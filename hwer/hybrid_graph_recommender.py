import time
from collections import Counter
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from bidict import bidict
from more_itertools import flatten
from sklearn.model_selection import StratifiedKFold
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from tensorflow import keras
from tensorflow.keras import layers

import networkx as nx
import tensorflow as tf
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data


from .svdpp_hybrid import SVDppHybrid
from .logging import getLogger
from .recommendation_base import EntityType
from .utils import RatingPredRegularization, get_rng, \
    LRSchedule, resnet_layer_with_content, ScaledGlorotNormal, root_mean_squared_error, mean_absolute_error, \
    normalize_affinity_scores_by_user_item_bs, get_clipped_rmse, unit_length_violations, UnitLengthRegularization


class GCNLayer(layers.Layer):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 gcn_msg,
                 gcn_reduce,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.gcn_msg = gcn_msg
        self.gcn_reduce = gcn_reduce

        # w_init = tf.random_normal_initializer()
        w_init = tf.initializers.glorot_uniform()
        # w_init = tf.initializers.TruncatedNormal()
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feats, out_feats),
                                                       dtype='float32'),
                                  trainable=True)
        if dropout:
            self.dropout = layers.Dropout(rate=dropout)
        else:
            self.dropout = 0.0
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,),
                                                         dtype='float32'),
                                    trainable=True)
        else:
            self.bias = None
        self.activation = activation

    def call(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = tf.matmul(h, self.weight)
        self.g.update_all(self.gcn_msg, self.gcn_reduce)
        h = self.g.ndata['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

    def __call__(self, h):
        return self.call(h)


class GCN(layers.Layer):
    def __init__(self,
                 g,
                 gcn_msg,
                 gcn_reduce,
                 in_features,
                 n_hidden,
                 out_features,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = []

        # input layer
        self.layers.append(
            GCNLayer(g, in_features, n_hidden, activation, dropout, gcn_msg, gcn_reduce))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(g, n_hidden, n_hidden, activation, dropout, gcn_msg, gcn_reduce))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, out_features, None, dropout, gcn_msg, gcn_reduce))

    def call(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


class HybridGCNRec(SVDppHybrid):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 content_only_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims, fast_inference)
        self.log = getLogger(type(self).__name__)
        self.content_only_inference = content_only_inference

    def __user_item_affinities_triplet_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_output_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
        self.log.debug("Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)

        def gcn_msg(edge):
            msg = edge.src['h'] * edge.src['norm']
            msg = msg * edge.data['weight']
            return {'m': msg}

        def gcn_reduce(node):
            accum = tf.reduce_sum(node.mailbox['m'], 1) * node.data['norm']
            accum = tf.math.l2_normalize(accum, axis=-1)
            return {'h': accum}

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        gcn_lr = hyperparams["gcn_lr"] if "gcn_lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 5
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 5
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.5
        random_pair_user_item_proba = hyperparams[
            "random_pair_user_item_proba"] if "random_pair_user_item_proba" in hyperparams else 0.4
        random_positive_weight = hyperparams[
            "random_positive_weight"] if "random_positive_weight" in hyperparams else 0.05
        random_negative_weight = hyperparams[
            "random_negative_weight"] if "random_negative_weight" in hyperparams else 0.25
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5


        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        max_affinity = np.max([r for u, i, r in user_item_affinities])
        min_affinity = np.min([r for u, i, r in user_item_affinities])
        user_item_affinities = [(u, i, (2 * (r - min_affinity) / (max_affinity - min_affinity)) - 1) for u, i, r in
                                user_item_affinities]

        n_input_dims = user_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        total_users = len(user_ids)
        total_items = len(item_ids)
        aff_range = np.max([r for u1, u2, r in user_item_affinities]) - np.min(
            [r for u1, u2, r in user_item_affinities])
        random_positive_weight = random_positive_weight * aff_range
        random_negative_weight = random_negative_weight * aff_range

        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i]) for u, i, r in user_item_affinities]
        src, dst = tuple(zip(*edge_list))
        g = DGLGraph()
        g.add_nodes(total_users + total_items)
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        g.add_edges(g.nodes(), g.nodes())
        n_edges = g.number_of_edges()
        # # normalization
        degs = g.in_degrees()
        degs = tf.cast(tf.identity(degs), dtype=tf.float32)
        norm = tf.math.pow(degs, -0.5)
        norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)
        g.ndata['norm'] = tf.expand_dims(norm, -1)
        ratings = [r for u, i, r in user_item_affinities]
        g.edata['weight'] = tf.expand_dims(np.array(ratings + ratings + list(np.ones(total_users + total_items))).astype(np.float32), -1)


        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            user_close_dict = defaultdict(list)
            user_far_dict = defaultdict(list)
            item_close_dict = defaultdict(list)
            item_far_dict = defaultdict(list)
            affinities = [(user_id_to_index[i], item_id_to_index[j], r) for i, j, r in affinities]
            for i, j, r in affinities:
                if r > 0:
                    user_close_dict[i].append((total_users + j, r))
                    item_close_dict[j].append((i, r))
                if r <= 0:
                    user_far_dict[i].append((total_users + j, r))
                    item_far_dict[j].append((i, r))

            def triplet_wt_fn(x):
                return 1 + 0.1 * np.log1p(np.abs(x / aff_range))

            def get_one_example(i, j, r):
                user = i
                second_item = total_users + j
                random_item = total_users + np.random.randint(0, total_items)
                random_user = np.random.randint(0, total_users)
                choose_random_pair = np.random.rand() < (random_pair_proba if r > 0 else random_pair_proba / 100)
                choose_user_pair = np.random.rand() < random_pair_user_item_proba
                if r < 0:
                    distant_item = second_item
                    distant_item_weight = r

                    if choose_random_pair or (i not in user_close_dict and j not in item_close_dict):
                        second_item, close_item_weight = random_user if choose_user_pair else random_item, random_positive_weight
                    else:
                        if (choose_user_pair and j in item_close_dict) or i not in user_close_dict:
                            second_item, close_item_weight = item_close_dict[j][
                                np.random.randint(0, len(item_close_dict[j]))]
                        else:
                            second_item, close_item_weight = user_close_dict[i][
                                np.random.randint(0, len(user_close_dict[i]))]
                else:
                    close_item_weight = r
                    if choose_random_pair or (i not in user_far_dict and j not in item_far_dict):
                        distant_item, distant_item_weight = random_user if choose_user_pair else random_item, random_negative_weight
                    else:
                        if (choose_user_pair and j in item_far_dict) or i not in user_far_dict:
                            distant_item, distant_item_weight = item_far_dict[j][
                                np.random.randint(0, len(item_far_dict[j]))]
                        else:
                            distant_item, distant_item_weight = user_far_dict[i][
                                np.random.randint(0, len(user_far_dict[i]))]

                close_item_weight = triplet_wt_fn(close_item_weight)
                distant_item_weight = triplet_wt_fn(distant_item_weight)
                return (user, second_item, distant_item, close_item_weight, distant_item_weight), 0

            def generator():
                for i in range(0, len(affinities), batch_size * 10):
                    start = i
                    end = min(i + batch_size * 10, len(affinities))
                    generated = [get_one_example(u, v, w) for u, v, w in affinities[start:end]]
                    for g in generated:
                        yield g

            return generator

        output_shapes = (((), (), (), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64, tf.float32, tf.float32), tf.float32)

        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size * 10).batch(batch_size).prefetch(32)

        def build_base_network(embedding_size, n_output_dims, vectors):
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(n_output_dims, activation="tanh", use_bias=False,
                                       kernel_initializer="glorot_uniform")
            item = dense(item)
            item = UnitLengthRegularization(l1=0.1)(item)
            # item = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(item)
            item = K.l2_normalize(item, axis=-1)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        bn = build_base_network(n_input_dims, n_output_dims, np.concatenate((user_vectors, item_vectors)))
        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))
        input_3 = keras.Input(shape=(1,))

        close_weight = keras.Input(shape=(1,))
        far_weight = keras.Input(shape=(1,))

        item_1 = bn(input_1)
        item_2 = bn(input_2)
        item_3 = bn(input_3)

        i1_i2_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
        i1_i2_dist = 1 - i1_i2_dist
        i1_i2_dist = close_weight * i1_i2_dist

        i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
        i1_i3_dist = 1 - i1_i3_dist
        i1_i3_dist = i1_i3_dist / K.abs(far_weight)

        loss = K.relu(i1_i2_dist - i1_i3_dist + margin)
        model = keras.Model(inputs=[input_1, input_2, input_3, close_weight, far_weight],
                            outputs=[loss])

        encoder = bn
        learning_rate = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=['mean_squared_error'], metrics=["mean_squared_error"])

        # model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        # user_vectors = encoder.predict(
        #     tf.data.Dataset.from_tensor_slices([user_id_to_index[i] for i in user_ids]).batch(batch_size).prefetch(16))
        # item_vectors = encoder.predict(
        #     tf.data.Dataset.from_tensor_slices([total_users + item_id_to_index[i] for i in item_ids]).batch(
        #         batch_size).prefetch(16))
        self.log.debug(
            "End Training User-Item Affinities, Unit Length Violations:: user = %s, item = %s, margin = %.4f",
            unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1), margin)

        features = np.concatenate((user_vectors, item_vectors))
        model = GCN(g,
                    gcn_msg,
                    gcn_reduce,
                    features.shape[1],
                    self.n_collaborative_dims * 2,
                    self.n_collaborative_dims,
                    gcn_layers,
                    tf.nn.leaky_relu,
                    0.0)
        learning_rate = LRSchedule(lr=gcn_lr, epochs=epochs, batch_size=len(user_item_affinities), n_examples=len(user_item_affinities))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=gcn_lr, decay=5e-4)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        loss = tf.keras.losses.MeanSquaredError()
        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size * 10).batch(batch_size).prefetch(32)
        for epoch in range(gcn_epochs):
            start = time.time()
            with tf.GradientTape() as tape:
                vectors = model(features)
                vectors = K.l2_normalize(vectors, axis=1)
                for x, y in train:

                    item_1 = tf.gather(vectors, x[0])
                    item_2 = tf.gather(vectors, x[1])
                    item_3 = tf.gather(vectors, x[2])


                    close_weight = x[3]
                    far_weight = x[4]

                    i1_i2_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
                    i1_i2_dist = 1 - i1_i2_dist
                    i1_i2_dist = close_weight * i1_i2_dist

                    i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
                    i1_i3_dist = 1 - i1_i3_dist
                    i1_i3_dist = i1_i3_dist / K.abs(far_weight)

                    error = K.relu(i1_i2_dist - i1_i3_dist + margin)
                    loss_value = loss(y, error)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            total_time = time.time() - start
            print("Epoch = %s/%s, Loss = %.4f, LR = %.6f, Time = %.1fs"%(epoch+1, gcn_epochs, loss_value, K.get_value(optimizer.learning_rate.lr), total_time))

        vectors = model(features)
        user_vectors, item_vectors = vectors[:total_users], vectors[total_users:]
        return user_vectors, item_vectors

