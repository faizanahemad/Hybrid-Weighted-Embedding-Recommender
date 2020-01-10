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
    normalize_affinity_scores_by_user_item_bs, get_clipped_rmse, unit_length_violations, UnitLengthRegularization, \
    unit_length


class GCNLayer(layers.Layer):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 gcn_msg,
                 gcn_reduce):
        super(GCNLayer, self).__init__()
        self.g = g
        self.gcn_msg = gcn_msg
        self.gcn_reduce = gcn_reduce
        self.gut = 0.0
        self.mmt1 = 0.0
        self.mmt2 = 0.0
        self.norm_time = 0.0

        # w_init = tf.random_normal_initializer()
        w_init = tf.initializers.glorot_uniform()
        # w_init = tf.initializers.TruncatedNormal(stddev=0.1)
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feats * 4, out_feats),
                                                       dtype='float32'), dtype=tf.float32,
                                  trainable=True)
        self.weight2 = tf.Variable(initial_value=w_init(shape=(out_feats, out_feats),
                                                       dtype='float32'), dtype=tf.float32,
                                  trainable=True)
        if dropout:
            self.dropout = layers.Dropout(rate=dropout)
        else:
            self.dropout = 0.0
        self.activation = activation

    def call(self, h):
        self.g.ndata['h'] = h

        self.g.ndata['norm_h'] = self.g.ndata['h'] * self.g.ndata['norm']
        start = time.time()
        self.g.update_all(fn.copy_src('norm_h', 'm'), fn.sum('m', 'agg_h'))  # consider fn.mean here
        end = time.time()
        self.gut += (end - start)
        self.g.ndata['agg_h'] = self.g.ndata['agg_h'] * self.g.ndata['norm']  # consider removing 'norm' if fn.mean used above
        h = tf.concat((h, self.g.ndata['agg_h']), axis=1)

        start = time.time()
        h = tf.matmul(h, self.weight)
        if self.activation:
            h = self.activation(h)
        end = time.time()
        self.mmt1 += (end - start)
        start = time.time()
        h = tf.math.l2_normalize(h, axis=-1)
        end = time.time()
        self.norm_time += (end - start)

        start = time.time()
        h = tf.matmul(h, self.weight2)
        if self.activation:
            h = self.activation(h)
        end = time.time()
        self.mmt2 += (end - start)

        if self.dropout:
            h = self.dropout(h)
        #
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
        self.g = g

        # input layer
        self.layers.append(
            GCNLayer(g, in_features+1, n_hidden, activation, dropout, gcn_msg, gcn_reduce))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(g, n_hidden, n_hidden, activation, dropout, gcn_msg, gcn_reduce))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, out_features, None, dropout, gcn_msg, gcn_reduce))

    def call(self, features):
        h = tf.concat((features, tf.reshape(self.g.ndata['degree'], (-1, 1))), axis=1)
        # h = features
        for layer in self.layers:
            h = layer(h)
        return h


class HybridGCNRec(SVDppHybrid):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims, fast_inference)
        self.log = getLogger(type(self).__name__)
        self.super_fast_inference = super_fast_inference

    def __build_user_item_graph__(self, edge_list: List[Tuple[int, int]], edge_weights: List[float], total_users: int, total_items: int):

        src, dst = tuple(zip(*edge_list))
        g = DGLGraph()
        g.add_nodes(total_users + total_items)

        g.add_edges(src, dst)
        g.add_edges(dst, src)
        n_edges = g.number_of_edges()
        # # normalization
        degs = g.in_degrees()
        degs = tf.cast(tf.identity(degs), dtype=tf.float32)
        norm = tf.math.pow(degs, -0.5)
        norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)
        g.ndata['norm'] = tf.expand_dims(norm, -1)
        g.ndata['degree'] = tf.math.tanh(tf.math.log1p(tf.math.log1p(tf.math.log1p(degs/100.0))))
        edge_weights = np.array(edge_weights + edge_weights).astype(np.float32)
        ew_min = np.min(edge_weights)
        ew_max = np.max(edge_weights)
        edge_weights = (edge_weights - ew_min)/(ew_max - ew_min)
        g.edata['weight'] = tf.expand_dims(edge_weights, -1)

        def gcn_msg(edge):
            msg = edge.src['h'] * edge.src['norm']
            msg = msg * edge.data['weight']
            return {'m': msg}

        def gcn_reduce(node):
            accum = tf.reduce_sum(node.mailbox['m'], 1) * node.data['norm']
            return {'h': accum}

        return g, gcn_msg, gcn_reduce

    def __user_item_affinities_triplet_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_output_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
        self.log.debug("Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        gcn_lr = hyperparams["gcn_lr"] if "gcn_lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 5
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 5
        gcn_batch_size = hyperparams["gcn_batch_size"] if "gcn_batch_size" in hyperparams else 5
        gcn_dropout = hyperparams["gcn_dropout"] if "gcn_dropout" in hyperparams else 0.0
        gcn_hidden_dims = hyperparams["gcn_hidden_dims"] if "gcn_hidden_dims" in hyperparams else self.n_collaborative_dims * 4
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        user_triplet_vectors, item_triplet_vectors = super().__user_item_affinities_triplet_trainer__(user_ids, item_ids, user_item_affinities,
                              user_vectors, item_vectors,
                              user_id_to_index,
                              item_id_to_index,
                              n_output_dims,
                              hyperparams)
        total_users = len(user_ids)
        total_items = len(item_ids)

        # user_vectors = np.concatenate((user_vectors, user_triplet_vectors), axis=1)
        # item_vectors = np.concatenate((item_vectors, item_triplet_vectors), axis=1)
        user_vectors = user_triplet_vectors
        item_vectors = item_triplet_vectors

        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i]) for u, i, r in user_item_affinities]
        ratings = [r for u, i, r in user_item_affinities]
        g, gcn_msg, gcn_reduce = self.__build_user_item_graph__(edge_list, ratings, total_users, total_items,)

        gcn_msg = fn.copy_src('norm_h', 'm')
        gcn_reduce = fn.sum('m', 'h')

        features = unit_length(np.concatenate((user_vectors, item_vectors)), axis=1)
        # embedding = tf.Variable(initial_value=features, dtype=tf.float32, trainable=True)
        model = GCN(g,
                    gcn_msg,
                    gcn_reduce,
                    features.shape[1],
                    gcn_hidden_dims,
                    self.n_collaborative_dims,
                    gcn_layers,
                    tf.nn.leaky_relu,
                    gcn_dropout)
        learning_rate = LRSchedule(lr=gcn_lr, epochs=gcn_epochs, batch_size=gcn_batch_size, n_examples=len(user_item_affinities))
        # optimizer = tf.keras.optimizers.Adam(learning_rate=gcn_lr, decay=5e-4)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        loss = tf.keras.losses.MeanSquaredError()
        train = self.__user_item_affinities_triplet_trainer_data_gen__(user_ids, item_ids,
                                                                       user_item_affinities,
                                                                       user_id_to_index, item_id_to_index,
                                                                       hyperparams)
        train = train.shuffle(gcn_batch_size * 10).batch(gcn_batch_size).prefetch(8)
        for epoch in range(gcn_epochs):
            start = time.time()
            total_rmse = 0.0
            total_loss = 0.0
            total_iters = 0
            for x, y in train:
                with tf.GradientTape() as tape:
                    vectors = model(features)
                    vectors = K.l2_normalize(vectors, axis=1)

                    item_1 = tf.gather(vectors, x[0])
                    item_2 = tf.gather(vectors, x[1])
                    item_3 = tf.gather(vectors, x[2])

                    i1_i2_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
                    i1_i2_dist = 1 - i1_i2_dist

                    i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
                    i1_i3_dist = 1 - i1_i3_dist

                    error = K.relu(i1_i2_dist - i1_i3_dist + margin)
                    loss_value = loss(y, error)
                    total_rmse = total_rmse + loss_value
                    loss_value += gcn_kernel_l2 * tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(v)) for v in model.trainable_weights])
                    total_loss += loss_value
                    total_iters += 1

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                del tape
            total_rmse = total_rmse/total_iters
            total_loss = total_loss / total_iters
            total_time = time.time() - start
            print("Epoch = %s/%s, RMSE = %.4f, Loss = %.6f, LR = %.6f, Time = %.1fs" % (epoch+1, gcn_epochs, total_rmse,total_loss , K.get_value(optimizer.learning_rate.lr), total_time))

        vectors = model(features)
        # K.l2_normalize(embedding, axis=1)
        user_vectors, item_vectors = vectors[:total_users], vectors[total_users:]
        return user_triplet_vectors, item_triplet_vectors

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        n_collaborative_dims = user_vectors.shape[1]
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 128
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        use_resnet = hyperparams["use_resnet"] if "use_resnet" in hyperparams else False

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # For unseen users and items creating 2 mock nodes
        user_content_vectors = np.concatenate((np.zeros((1, user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1, item_content_vectors.shape[1])), item_content_vectors))
        user_vectors = np.concatenate((np.zeros((1, user_vectors.shape[1])), user_vectors))
        item_vectors = np.concatenate((np.zeros((1, item_vectors.shape[1])), item_vectors))

        mu, user_bias, item_bias, train, \
        ratings_count_by_user, ratings_count_by_item, \
        min_affinity, \
        max_affinity, user_item_list, item_user_list, \
        gen_fn, prediction_output_shape, prediction_output_types = self.__build_dataset__(user_ids, item_ids,
                                                                              user_item_affinities,
                                                                              user_content_vectors,
                                                                              item_content_vectors,
                                                                              user_vectors, item_vectors,
                                                                              user_id_to_index,
                                                                              item_id_to_index,
                                                                              rating_scale, hyperparams)
        assert np.sum(np.isnan(user_bias)) == 0
        assert np.sum(np.isnan(item_bias)) == 0
        assert np.sum(np.isnan(user_content_vectors)) == 0
        assert np.sum(np.isnan(item_content_vectors)) == 0
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0


        #
        lr = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities), divisor=5)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        loss = root_mean_squared_error
        total_users = len(user_ids) + 1
        total_items = len(item_ids) + 1
        edge_list = [(user_id_to_index[u]+1, total_users + item_id_to_index[i]+1) for u, i, r in user_item_affinities]
        ratings = [r for u, i, r in user_item_affinities]
        g, gcn_msg, gcn_reduce = self.__build_user_item_graph__(edge_list, ratings, total_users, total_items)
        # gcn_msg = fn.src_mul_edge('norm_h', "weight", 'm')
        gcn_msg = fn.copy_src('norm_h', 'm')
        gcn_reduce = fn.sum('m', 'h')
        if use_content:
            user_vectors = np.concatenate((user_vectors, user_content_vectors), axis=1)
            item_vectors = np.concatenate((item_vectors, item_content_vectors), axis=1)

        features = np.concatenate((user_vectors, item_vectors))
        features = unit_length(features, axis=1)
        features[np.isnan(features)] = 0.0
        features[np.isinf(features)] = 0.0
        user_bias = tf.Variable(initial_value=user_bias, dtype=tf.float32, trainable=True)
        item_bias = tf.Variable(initial_value=item_bias, dtype=tf.float32, trainable=True)
        w_init = tf.initializers.TruncatedNormal(stddev=0.01,)
        # features = tf.Variable(initial_value=features, dtype=tf.float32, trainable=True)
        user_item_dot_scaler = tf.Variable(initial_value=w_init(shape=(self.n_collaborative_dims, self.n_collaborative_dims),
                                                       dtype='float32'), dtype=tf.float32, trainable=True)
        item_items_dot_scaler = tf.Variable(
            initial_value=w_init(shape=(self.n_collaborative_dims, self.n_collaborative_dims),
                                 dtype='float32'), dtype=tf.float32, trainable=True)
        user_users_dot_scaler = tf.Variable(
            initial_value=w_init(shape=(self.n_collaborative_dims, self.n_collaborative_dims),
                                 dtype='float32'), dtype=tf.float32, trainable=True)

        model = GCN(g,
                    gcn_msg,
                    gcn_reduce,
                    features.shape[1],
                    network_width,
                    self.n_collaborative_dims,
                    network_depth,
                    tf.nn.leaky_relu,
                    dropout)
        # print("Before = ", len(model.trainable_weights),[tf.reduce_mean(tf.nn.l2_loss(v)).numpy() for v in model.trainable_weights])

        @tf.function
        def train_batch(x, y):
            with tf.GradientTape() as tape:
                vectors = model(features)
                user_input = x[0]
                item_input = x[1] + total_users
                users_input = x[2]
                items_input = x[3] + total_users
                nu_input = tf.expand_dims(tf.dtypes.cast(x[4], dtype=tf.float32), -1)
                ni_input = tf.expand_dims(tf.dtypes.cast(x[5], dtype=tf.float32), -1)

                user_vec = tf.gather(vectors, user_input)
                item_vec = tf.gather(vectors, item_input)

                users_vecs = tf.gather(vectors, users_input)
                items_vecs = tf.gather(vectors, items_input)
                users_vecs = tf.reduce_sum(users_vecs, axis=1)
                items_vecs = tf.reduce_sum(items_vecs, axis=1)
                users_vecs = tf.multiply(users_vecs, ni_input)
                items_vecs = tf.multiply(items_vecs, nu_input)

                user_item_vec_dot = tf.reduce_sum(
                    tf.multiply(tf.linalg.matmul(user_vec, user_item_dot_scaler), item_vec), axis=1)
                item_items_vec_dot = tf.reduce_sum(
                    tf.multiply(tf.linalg.matmul(item_vec, item_items_dot_scaler), items_vecs), axis=1)
                user_users_vec_dot = tf.reduce_sum(
                    tf.multiply(tf.linalg.matmul(user_vec, user_users_dot_scaler), users_vecs), axis=1)
                implicit_term = user_item_vec_dot + item_items_vec_dot + user_users_vec_dot
                bu = tf.gather(user_bias, user_input)
                bi = tf.gather(item_bias, x[1])
                bias_loss = bias_regularizer * (tf.reduce_mean(tf.nn.l2_loss(bu)) + tf.reduce_mean(tf.nn.l2_loss(bi)))
                kernel_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(v)) for v in model.trainable_weights])
                scaler_loss = tf.reduce_mean(tf.nn.l2_loss(user_item_dot_scaler))
                scaler_loss += tf.reduce_mean(tf.nn.l2_loss(item_items_dot_scaler))
                scaler_loss += tf.reduce_mean(tf.nn.l2_loss(user_users_dot_scaler))
                kernel_loss = kernel_l2 * kernel_loss
                scaler_loss = kernel_l2 * scaler_loss
                base_estimates = mu + bu + bi
                rating = base_estimates + implicit_term
                y = tf.dtypes.cast(y, tf.float32)
                loss_value = loss(y, rating)
                rmse = loss_value
                loss_value = loss_value + bias_loss + kernel_loss + scaler_loss

            grads = tape.gradient(loss_value,
                                  list(model.trainable_weights) + [user_item_dot_scaler, item_items_dot_scaler,
                                                                   user_users_dot_scaler])

            optimizer.apply_gradients(zip(grads,
                                          list(model.trainable_weights) + [user_item_dot_scaler, item_items_dot_scaler,
                                                                           user_users_dot_scaler]))
            del tape
            return rmse, loss_value

        for epoch in range(0):
            start = time.time()
            total_iters = 0
            total_rmse = 0.0
            total_loss = 0.0
            for x, y in train:
                rmse, loss = train_batch(x, y)
                total_loss += loss
                total_rmse += rmse
                total_iters += 1
            total_time = time.time() - start
            total_rmse = total_rmse / total_iters
            total_loss = total_loss / total_iters
            print("Epoch = %s/%s, RMSE = %.4f, Loss = %.6f, LR = %.6f, Time = %.1fs" % (
                epoch + 1, epochs, total_rmse, total_loss, K.get_value(optimizer.learning_rate.lr), total_time))

        for epoch in range(epochs):
            start = time.time()
            total_iters = 0
            total_rmse = 0.0
            total_loss = 0.0
            non_graph_time = 0.0
            grad_time = 0.0
            ap_grad_time = 0.0
            for x, y in train:
                with tf.GradientTape() as tape:
                    vectors = model(features)
                    start2 = time.time()
                    user_input = x[0]
                    item_input = x[1] + total_users
                    users_input = x[2]
                    items_input = x[3] + total_users
                    nu_input = tf.expand_dims(tf.dtypes.cast(x[4], dtype=tf.float32), -1)
                    ni_input = tf.expand_dims(tf.dtypes.cast(x[5], dtype=tf.float32), -1)

                    user_vec = tf.gather(vectors, user_input)
                    item_vec = tf.gather(vectors, item_input)

                    users_vecs = tf.gather(vectors, users_input)
                    items_vecs = tf.gather(vectors, items_input)
                    users_vecs = tf.reduce_sum(users_vecs, axis=1)
                    items_vecs = tf.reduce_sum(items_vecs, axis=1)
                    users_vecs = tf.multiply(users_vecs, ni_input)
                    items_vecs = tf.multiply(items_vecs, nu_input)

                    user_item_vec_dot = tf.reduce_sum(tf.multiply(tf.linalg.matmul(user_vec, user_item_dot_scaler), item_vec), axis=1)
                    item_items_vec_dot = tf.reduce_sum(tf.multiply(tf.linalg.matmul(item_vec, item_items_dot_scaler), items_vecs), axis=1)
                    user_users_vec_dot = tf.reduce_sum(tf.multiply(tf.linalg.matmul(user_vec, user_users_dot_scaler), users_vecs), axis=1)
                    implicit_term = user_item_vec_dot + item_items_vec_dot + user_users_vec_dot
                    bu = tf.gather(user_bias, user_input)
                    bi = tf.gather(item_bias, x[1])
                    bias_loss = bias_regularizer * (tf.reduce_mean(tf.nn.l2_loss(bu)) + tf.reduce_mean(tf.nn.l2_loss(bi)))
                    kernel_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(v)) for v in model.trainable_weights])
                    scaler_loss = tf.reduce_mean(tf.nn.l2_loss(user_item_dot_scaler))
                    scaler_loss += tf.reduce_mean(tf.nn.l2_loss(item_items_dot_scaler))
                    scaler_loss += tf.reduce_mean(tf.nn.l2_loss(user_users_dot_scaler))
                    kernel_loss = kernel_l2 * kernel_loss
                    scaler_loss = kernel_l2 * scaler_loss
                    base_estimates = mu + bu + bi
                    rating = base_estimates + implicit_term
                    y = tf.dtypes.cast(y, tf.float32)
                    non_graph_time += (time.time() - start2)
                    loss_value = loss(y, rating)
                    total_rmse = total_rmse + loss_value
                    loss_value = loss_value + bias_loss + kernel_loss + scaler_loss
                    total_loss = total_loss + loss_value
                    # print("Nans = ", np.sum(np.isnan(vectors.numpy())), "Inf = ", np.sum(np.isinf(vectors.numpy())))
                    # print("Nans = ", np.sum(np.isnan(user_vec.numpy())))
                    # print("Nans = ", np.sum(np.isnan(item_vec.numpy())))
                    # print("Nans = ", np.sum(np.isnan(items_vecs.numpy())))
                    # print("Nans = ", np.sum(np.isnan(users_vecs.numpy())))
                    # print("Nans = ", np.sum(np.isnan(implicit_term.numpy())))
                    # print("Nans = ", np.sum(np.isnan(base_estimates.numpy())))
                    # print("Nans = ", np.sum(np.isnan(rating.numpy())))
                    # print("Nans = ", np.sum(np.isnan(loss_value.numpy())), " Loss = ", loss_value.numpy())
                    # print("Bias Loss = ", bias_loss.numpy(), "Scaler Loss = ", scaler_loss.numpy(), "Kernel Loss = ", kernel_loss.numpy(), "Total Loss = ", loss_value.numpy())
                    # print("=" * 80)

                    total_iters += 1
                gs = time.time()
                grads = tape.gradient(loss_value, list(model.trainable_weights) + [user_item_dot_scaler, item_items_dot_scaler, user_users_dot_scaler])
                grad_time += (time.time() - gs)
                gs = time.time()
                optimizer.apply_gradients(zip(grads, list(model.trainable_weights) + [user_item_dot_scaler, item_items_dot_scaler, user_users_dot_scaler]))
                del tape
                ap_grad_time += (time.time() - gs)


            # print("After = ", len(model.trainable_weights), [tf.reduce_mean(tf.nn.l2_loss(v)).numpy() for v in model.trainable_weights])
            total_time = time.time() - start
            total_rmse = total_rmse / total_iters
            total_loss = total_loss / total_iters
            graph_update_time = model.layers[0].gut
            norm_time = model.layers[0].norm_time
            mmt1 = model.layers[0].mmt1
            mmt2 = model.layers[0].mmt2
            print("Time: NGT = %.2f, Graph = %.2f, Mul-1 = %.2f, Mul-2 = %.2f, Norm = %.2f, Grad = %.2f, Apply Grad = %.2f" %
                  (non_graph_time, graph_update_time, mmt1, mmt2, norm_time, grad_time, ap_grad_time))
            print("Epoch = %s/%s, RMSE = %.4f, Loss = %.6f, LR = %.6f, Time = %.1fs" % (
            epoch + 1, epochs, total_rmse, total_loss, K.get_value(optimizer.learning_rate.lr), total_time))

        #
        prediction_artifacts = {"vectors": model(features), "user_item_list": user_item_list,
                                "item_user_list": item_user_list, "mu": mu, "user_bias": user_bias.numpy(), "item_bias": item_bias.numpy(),
                                "total_users": total_users, "user_item_dot_scaler": user_item_dot_scaler.numpy(),
                                "user_users_dot_scaler": user_users_dot_scaler.numpy(), "item_items_dot_scaler": item_items_dot_scaler.numpy(),
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size, "gen_fn": gen_fn}
        self.log.info("Built Prediction Network, model params = %s", model.count_params())
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        start = time.time()
        vectors = self.prediction_artifacts["vectors"]
        mu = self.prediction_artifacts["mu"]
        user_bias = self.prediction_artifacts["user_bias"]
        item_bias = self.prediction_artifacts["item_bias"]
        total_users = self.prediction_artifacts["total_users"]
        user_item_dot_scaler = self.prediction_artifacts["user_item_dot_scaler"]
        user_users_dot_scaler = self.prediction_artifacts["user_users_dot_scaler"]
        item_items_dot_scaler = self.prediction_artifacts["item_items_dot_scaler"]


        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]
        batch_size = self.prediction_artifacts["batch_size"]
        gen_fn = self.prediction_artifacts["gen_fn"]
        batch_size = max(512, batch_size)

        def generate_prediction_samples(affinities):
            def generator():
                for i in range(0, len(affinities), batch_size):
                    start = i
                    end = min(i + batch_size, len(affinities))
                    generated = np.array([gen_fn(u, v, nu, ni) for u, v, nu, ni in affinities[start:end]])
                    yield generated
            return generator

        if self.fast_inference:
            return self.fast_predict(user_item_pairs)

        if self.super_fast_inference:
            assert self.mu is not None
            return [self.mu + self.bu[u] + self.bi[i] for u, i, in user_item_pairs]

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0,
                ratings_count_by_user[self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0],
                ratings_count_by_item[self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0]) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0

        predictions = []
        for x in generate_prediction_samples(uip)():
            user_input = x[:, 0].astype(int)
            item_input = x[:, 1].astype(int) + total_users
            users_input = np.array([np.array(a, dtype=int) for a in x[:, 2]])
            items_input = np.array([np.array(a, dtype=int) for a in x[:, 3]]) + total_users
            nu_input = np.expand_dims(x[:, 4].astype(float), -1)
            ni_input = np.expand_dims(x[:, 5].astype(float), -1)

            user_vec = np.take(vectors, user_input, axis=0)
            item_vec = np.take(vectors, item_input, axis=0)

            users_vecs = np.take(vectors, users_input, axis=0)
            items_vecs = np.take(vectors, items_input, axis=0)

            users_vecs = np.sum(users_vecs, axis=1)
            items_vecs = np.sum(items_vecs, axis=1)
            users_vecs = np.multiply(users_vecs, ni_input)
            items_vecs = np.multiply(items_vecs, nu_input)

            user_item_vec_dot = np.sum(np.multiply(np.matmul(user_vec, user_item_dot_scaler), item_vec), axis=1)
            item_items_vec_dot = np.sum(np.multiply(np.matmul(item_vec, item_items_dot_scaler), items_vecs), axis=1)
            user_users_vec_dot = np.sum(np.multiply(np.matmul(user_vec, user_users_dot_scaler), users_vecs), axis=1)
            implicit_term = user_item_vec_dot + item_items_vec_dot + user_users_vec_dot
            bu = np.take(user_bias, user_input, axis=0)
            bi = np.take(item_bias, x[:, 1].astype(int), axis=0)
            rating = mu + bu + bi + implicit_term
            predictions.extend(rating)

        predictions = np.array(predictions)
        assert np.sum(np.isnan(predictions)) == 0
        predictions[np.isnan(predictions)] = [self.mu + self.bu[u] + self.bi[i] for u, i in np.array(user_item_pairs)[np.isnan(predictions)]]
        assert len(predictions) == len(user_item_pairs)
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        self.log.info("Finished Predicting for n_samples = %s, time taken = %.2f",
                      len(user_item_pairs),
                      time.time() - start)
        return predictions
