import abc
import operator
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

from .content_recommender import ContentRecommendation
from .logging import getLogger
from .recommendation_base import EntityType
from .recommendation_base import RecommendationBase, FeatureSet
from .utils import unit_length, normalize_affinity_scores_by_user, UnitLengthRegularization, unit_length_violations, \
    LRSchedule, \
    resnet_layer_with_content


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_output_dims=n_content_dims + n_collaborative_dims)
        self.cb = ContentRecommendation(embedding_mapper, knn_params, rating_scale,
                                        n_content_dims, )
        self.n_content_dims = n_content_dims
        self.n_collaborative_dims = n_collaborative_dims
        self.content_data_used = None
        self.prediction_artifacts = None
        self.log = getLogger(type(self).__name__)

    def __entity_entity_affinities_trainer__(self,
                                             entity_ids: List[str],
                                             entity_entity_affinities: List[Tuple[str, str, float]],
                                             entity_id_to_index: Dict[str, int],
                                             vectors: np.ndarray,
                                             n_output_dims: int,
                                             hyperparams: Dict) -> np.ndarray:
        self.log.debug("Start Training Entity Affinities, n_entities = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(entity_ids), len(entity_entity_affinities), vectors.shape, n_output_dims)
        ratings = np.array([r for u, i, r in entity_entity_affinities])
        min_affinity = np.min(ratings)
        max_affinity = np.max(ratings)
        entity_entity_affinities = [(u, i, (2 * 0.8 * (r - min_affinity) / (max_affinity - min_affinity)) - 0.8) for u, i, r in
                                    entity_entity_affinities]
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            def generator():
                for i, j, r in affinities:
                    first_item = entity_id_to_index[i]
                    second_item = entity_id_to_index[j]
                    r = np.clip(r, -1.0, 1.0)
                    yield (first_item, second_item), r

            return generator

        output_shapes = (((), ()), ())
        output_types = ((tf.int64, tf.int64), tf.float32)

        train = tf.data.Dataset.from_generator(generate_training_samples(entity_entity_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        train = train.shuffle(batch_size*10).batch(batch_size).prefetch(32)

        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))

        def build_base_network(embedding_size, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(entity_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(embedding_size, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
            item = dense(item)
            item = UnitLengthRegularization(l1=0.1)(item)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        bn = build_base_network(n_output_dims, vectors)

        item_1 = bn(input_1)
        item_2 = bn(input_2)

        pred = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
        pred = K.tanh(pred)
        model = keras.Model(inputs=[input_1, input_2],
                            outputs=[pred])
        encoder = bn

        learning_rate = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(entity_entity_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=['mean_squared_error'], metrics=["mean_squared_error"])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([entity_id_to_index[i] for i in entity_ids]).batch(batch_size).prefetch(16))
        self.log.debug("End Training Entity Affinities, Unit Length Violations = %s",
                       unit_length_violations(vectors, axis=1))
        return vectors

    def __entity_entity_affinities_triplet_trainer__(self,
                                             entity_ids: List[str],
                                             entity_entity_affinities: List[Tuple[str, str, float]],
                                             entity_id_to_index: Dict[str, int],
                                             vectors: np.ndarray,
                                             n_output_dims: int,
                                             hyperparams: Dict) -> np.ndarray:
        self.log.debug("Start Training Entity Affinities, n_entities = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(entity_ids), len(entity_entity_affinities), vectors.shape, n_output_dims)
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.2
        random_positive_weight = hyperparams[
            "random_positive_weight"] if "random_positive_weight" in hyperparams else 0.05
        random_negative_weight = hyperparams[
            "random_negative_weight"] if "random_negative_weight" in hyperparams else 0.2
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.1

        aff_range = np.max([r for u1, u2, r in entity_entity_affinities]) - np.min([r for u1, u2, r in entity_entity_affinities])
        random_positive_weight = random_positive_weight * aff_range
        random_negative_weight = random_negative_weight * aff_range

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            item_close_dict = {}
            item_far_dict = {}
            for i, j, r in affinities:
                assert r != 0
                if r > 0:
                    if i in item_close_dict:
                        item_close_dict[i].append((j, r))
                    else:
                        item_close_dict[i] = [(j, r)]

                    if j in item_close_dict:
                        item_close_dict[j].append((i, r))
                    else:
                        item_close_dict[j] = [(i, r)]
                if r < 0:
                    if i in item_far_dict:
                        item_far_dict[i].append((j, r))
                    else:
                        item_far_dict[i] = [(j, r)]

                    if j in item_far_dict:
                        item_far_dict[j].append((i, r))
                    else:
                        item_far_dict[j] = [(i, r)]
            total_items = len(entity_ids)
            triplet_wt_fn = lambda x: 1 + np.log1p(np.abs(x/aff_range))

            def generator():
                for i, j, r in affinities:
                    first_item = entity_id_to_index[i]
                    second_item = entity_id_to_index[j]
                    random_item = entity_id_to_index[entity_ids[np.random.randint(0, total_items)]]
                    choose_random_pair = np.random.rand() < (random_pair_proba if r > 0 else random_pair_proba/10)
                    if r < 0:
                        distant_item = second_item
                        distant_item_weight = r

                        if choose_random_pair or i not in item_close_dict:
                            second_item, close_item_weight = random_item, random_positive_weight

                        else:
                            second_item, close_item_weight = item_close_dict[i][
                                np.random.randint(0, len(item_close_dict[i]))]
                            second_item = entity_id_to_index[second_item]
                    else:
                        close_item_weight = r
                        if choose_random_pair or i not in item_far_dict:
                            distant_item, distant_item_weight = random_item, random_negative_weight

                        else:
                            distant_item, distant_item_weight = item_far_dict[i][
                                np.random.randint(0, len(item_far_dict[i]))]
                            distant_item = entity_id_to_index[distant_item]
                    close_item_weight = triplet_wt_fn(close_item_weight)
                    distant_item_weight = triplet_wt_fn(distant_item_weight)
                    yield (first_item, second_item, distant_item, close_item_weight, distant_item_weight), 0

            return generator

        output_shapes = (((), (), (), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64, tf.float32, tf.float32), tf.float32)

        train = tf.data.Dataset.from_generator(generate_training_samples(entity_entity_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        train = train.shuffle(batch_size*10).batch(batch_size).prefetch(32)

        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))
        input_3 = keras.Input(shape=(1,))

        close_weight = keras.Input(shape=(1,))
        far_weight = keras.Input(shape=(1,))

        def build_base_network(embedding_size, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(entity_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(embedding_size, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
            item = dense(item)
            item = UnitLengthRegularization(l1=0.1)(item)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        bn = build_base_network(n_output_dims, vectors)

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
        learning_rate = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(entity_entity_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=['mean_squared_error'], metrics=["mean_squared_error"])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([entity_id_to_index[i] for i in entity_ids]).batch(batch_size).prefetch(
                16))
        self.log.debug("End Training Entity Affinities, Unit Length Violations = %s", unit_length_violations(vectors, axis=1))
        return vectors

    def __build_collaborative_embeddings__(self, user_item_affinities: List[Tuple[str, str, float]],
                                           item_item_affinities: List[Tuple[str, str, bool]],
                                           user_user_affinities: List[Tuple[str, str, bool]],
                                           user_ids: List[str], item_ids: List[str],
                                           user_vectors: np.ndarray, item_vectors: np.ndarray,
                                           hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:

        use_triplet = hyperparams["use_triplet"] if "use_triplet" in hyperparams else False
        entity_affiity_fn = self.__entity_entity_affinities_trainer__
        user_item_affinity_fn = self.__user_item_affinities_trainer__
        if use_triplet:
            self.log.debug("Triplet Loss Based Training")
            entity_affiity_fn = self.__entity_entity_affinities_triplet_trainer__
            user_item_affinity_fn = self.__user_item_affinities_triplet_trainer__

        if len(item_item_affinities) > 0:
            item_item_params = {} if "item_item_params" not in hyperparams else hyperparams["item_item_params"]

            item_vectors = entity_affiity_fn(entity_ids=item_ids,
                                                                     entity_entity_affinities=item_item_affinities,
                                                                     entity_id_to_index=self.item_id_to_index,
                                                                     vectors=item_vectors,
                                                                     n_output_dims=self.n_content_dims,
                                                                     hyperparams=item_item_params)

        if len(user_user_affinities) > 0:
            user_user_params = {} if "user_user_params" not in hyperparams else hyperparams["user_user_params"]
            user_vectors = entity_affiity_fn(entity_ids=user_ids,
                                                                     entity_entity_affinities=user_user_affinities,
                                                                     entity_id_to_index=self.user_id_to_index,
                                                                     vectors=user_vectors,
                                                                     n_output_dims=self.n_content_dims,
                                                                     hyperparams=user_user_params)

        if len(user_item_affinities) > 0:
            user_item_params = {} if "user_item_params" not in hyperparams else hyperparams["user_item_params"]
            user_vectors, item_vectors = user_item_affinity_fn(user_ids, item_ids, user_item_affinities,
                                                                               user_vectors, item_vectors,
                                                                               self.user_id_to_index,
                                                                               self.item_id_to_index,
                                                                               self.n_collaborative_dims,
                                                                               user_item_params)
        self.log.info("Built Collaborative Embeddings, user_vectors shape = %s, item_vectors shape = %s",
                      user_vectors.shape, item_vectors.shape)
        return user_vectors, item_vectors

    def __user_item_affinities_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_output_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
        self.log.debug(
            "Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
            len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001

        # max_affinity = np.max(np.abs([r for u, i, r in user_item_affinities]))
        max_affinity = np.max([r for u, i, r in user_item_affinities])
        min_affinity = np.min([r for u, i, r in user_item_affinities])
        user_item_affinities = [(u, i, (2 * 0.9 * (r - min_affinity) / (max_affinity - min_affinity)) - 0.9) for u, i, r
                                in
                                user_item_affinities]

        n_input_dims = user_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        total_users = len(user_ids)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            def generator():
                for i, j, r in affinities:
                    user = user_id_to_index[i]
                    item = total_users + item_id_to_index[j]
                    yield (user, item), r

            return generator

        output_shapes = (((), ()), ())
        output_types = ((tf.int64, tf.int64), tf.float32)
        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size*10).batch(batch_size)

        def build_base_network(embedding_size, n_output_dims, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            #
            dense = keras.layers.Dense(n_output_dims, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
            item = dense(item)

            item = UnitLengthRegularization(l1=0.1)(item)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))

        bn = build_base_network(n_input_dims, n_output_dims, np.concatenate((user_vectors, item_vectors)))
        user = bn(input_1)
        item = bn(input_2)

        pred = tf.keras.layers.Dot(axes=1, normalize=True)([user, item])
        pred = K.tanh(pred)

        model = keras.Model(inputs=[input_1, input_2],
                            outputs=[pred])
        encoder = bn
        learning_rate = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=['mean_squared_error'], metrics=["mean_squared_error"])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        user_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([user_id_to_index[i] for i in user_ids]).batch(batch_size).prefetch(16))
        item_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([total_users + item_id_to_index[i] for i in item_ids]).batch(batch_size).prefetch(16))
        self.log.debug("End Training User-Item Affinities, Unit Length Violations:: user = %s, item = %s",
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))
        return user_vectors, item_vectors

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
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.4
        random_pair_user_item_proba = hyperparams["random_pair_user_item_proba"] if "random_pair_user_item_proba" in hyperparams else 0.4
        random_positive_weight = hyperparams["random_positive_weight"] if "random_positive_weight" in hyperparams else 0.25
        random_negative_weight = hyperparams["random_negative_weight"] if "random_negative_weight" in hyperparams else 0.5
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5

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

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            user_close_dict = defaultdict(list)
            user_far_dict = defaultdict(list)
            item_close_dict = defaultdict(list)
            item_far_dict = defaultdict(list)
            for i, j, r in affinities:
                assert r != 0
                if r > 0:
                    user_close_dict[i].append((j, r))
                    item_close_dict[j].append((i, r))
                if r < 0:
                    user_far_dict[i].append((j, r))
                    item_far_dict[j].append((i, r))

            triplet_wt_fn = lambda x: 1 + np.log1p(np.abs(x / aff_range))

            def generator():
                for i, j, r in affinities:
                    user = user_id_to_index[i]
                    second_item = total_users + item_id_to_index[j]
                    random_item = total_users + item_id_to_index[item_ids[np.random.randint(0, total_items)]]
                    random_user = user_id_to_index[user_ids[np.random.randint(0, total_users)]]
                    choose_random_pair = np.random.rand() < (random_pair_proba if r > 0 else random_pair_proba/10)
                    choose_user_pair = np.random.rand() < random_pair_user_item_proba
                    if r < 0:
                        distant_item = second_item
                        distant_item_weight = r

                        if choose_random_pair or (i not in user_close_dict and j not in item_close_dict):
                            second_item, close_item_weight = random_user if choose_user_pair else random_item, random_positive_weight
                        else:
                            if (choose_user_pair and j in item_close_dict) or i not in user_close_dict:
                                second_item, close_item_weight = item_close_dict[j][np.random.randint(0, len(item_close_dict[j]))]
                                second_item = user_id_to_index[second_item]
                            else:
                                second_item, close_item_weight = user_close_dict[i][np.random.randint(0, len(user_close_dict[i]))]
                                second_item = total_users + item_id_to_index[second_item]
                    else:
                        close_item_weight = r
                        if choose_random_pair or (i not in user_far_dict and j not in item_far_dict):
                            distant_item, distant_item_weight = random_user if choose_user_pair else random_item, random_negative_weight
                        else:
                            if (choose_user_pair and j in item_far_dict) or i not in user_far_dict:
                                distant_item, distant_item_weight = item_far_dict[j][np.random.randint(0, len(item_far_dict[j]))]
                                distant_item = user_id_to_index[distant_item]
                            else:
                                distant_item, distant_item_weight = user_far_dict[i][np.random.randint(0, len(user_far_dict[i]))]
                                distant_item = total_users + item_id_to_index[distant_item]

                    close_item_weight = triplet_wt_fn(close_item_weight)
                    distant_item_weight = triplet_wt_fn(distant_item_weight)
                    yield (user, second_item, distant_item, close_item_weight, distant_item_weight), 0
            return generator

        output_shapes = (((), (), (), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64, tf.float32, tf.float32), tf.float32)

        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size*10).batch(batch_size).prefetch(32)

        def build_base_network(embedding_size, n_output_dims, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(n_output_dims, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
            item = dense(item)
            item = UnitLengthRegularization(l1=0.1)(item)
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

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        user_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([user_id_to_index[i] for i in user_ids]).batch(batch_size).prefetch(16))
        item_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([total_users + item_id_to_index[i] for i in item_ids]).batch(batch_size).prefetch(16))
        self.log.debug("End Training User-Item Affinities, Unit Length Violations:: user = %s, item = %s, margin = %.4f",
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1), margin)
        return user_vectors, item_vectors

    @abc.abstractmethod
    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):

        pass

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_item_affinities: List[Tuple[str, str, float]],
            **kwargs):
        start_time = time.time()
        _ = super().fit(user_ids, item_ids, user_item_affinities, **kwargs)
        _, _, _, _, user_normalized_affinities = normalize_affinity_scores_by_user(user_item_affinities)

        item_data: FeatureSet = kwargs["item_data"] if "item_data" in kwargs else FeatureSet([])
        user_data: FeatureSet = kwargs["user_data"] if "user_data" in kwargs else FeatureSet([])
        hyperparameters = {} if "hyperparameters" not in kwargs else kwargs["hyperparameters"]

        combining_factor: int = hyperparameters["combining_factor"] if "combining_factor" in hyperparameters else 0.5
        alpha = combining_factor
        assert 0 <= alpha <= 1
        content_data_used = ("item_data" in kwargs or "user_data" in kwargs) and alpha > 0
        self.content_data_used = content_data_used

        self.n_output_dims = self.n_content_dims + self.n_collaborative_dims if content_data_used else self.n_collaborative_dims

        item_item_affinities: List[Tuple[str, str, bool]] = kwargs[
            "item_item_affinities"] if "item_item_affinities" in kwargs else list()
        user_user_affinities: List[Tuple[str, str, bool]] = kwargs[
            "user_user_affinities"] if "user_user_affinities" in kwargs else list()

        if content_data_used:
            super(type(self.cb), self.cb).fit(user_ids, item_ids, user_item_affinities, **kwargs)
            user_vectors, item_vectors = self.cb.__build_content_embeddings__(user_ids, item_ids,
                                                                              user_data, item_data,
                                                                              user_normalized_affinities,
                                                                              self.n_content_dims)
        else:
            user_vectors, item_vectors = np.random.rand((len(user_ids), self.n_content_dims)), np.random.rand(
                (len(item_ids), self.n_content_dims))
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)

        user_content_vectors, item_content_vectors = user_vectors.copy(), item_vectors.copy()
        assert user_content_vectors.shape[1] == item_content_vectors.shape[1] == self.n_content_dims

        collaborative_params = {} if "collaborative_params" not in hyperparameters else hyperparameters[
            "collaborative_params"]
        user_vectors, item_vectors = self.__build_collaborative_embeddings__(user_normalized_affinities,
                                                                             item_item_affinities,
                                                                             user_user_affinities, user_ids, item_ids,
                                                                             user_vectors, item_vectors,
                                                                             collaborative_params)

        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)

        self.log.debug("Fit Method, Unit Length Violations:: user_content = %s, item_content = %s" +
                       "user_collab = %s, item_collab = %s",
                       unit_length_violations(user_content_vectors, axis=1), unit_length_violations(item_content_vectors, axis=1),
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))

        user_content_vectors, item_content_vectors = user_content_vectors * alpha, item_content_vectors * alpha
        user_vectors, item_vectors = user_vectors * (1 - alpha), item_vectors * (1 - alpha)
        assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_collaborative_dims
        prediction_network_params = {} if "prediction_network_params" not in collaborative_params else \
        collaborative_params[
            "prediction_network_params"]
        if content_data_used:

            prediction_artifacts = self.__build_prediction_network__(user_ids, item_ids, user_item_affinities,
                                                                     user_content_vectors, item_content_vectors,
                                                                     user_vectors, item_vectors,
                                                                     self.user_id_to_index, self.item_id_to_index,
                                                                     self.rating_scale, prediction_network_params)
        else:
            prediction_artifacts = self.__build_prediction_network__(user_ids, item_ids, user_item_affinities,
                                                                     user_vectors, item_vectors,
                                                                     user_vectors, item_vectors,
                                                                     self.user_id_to_index, self.item_id_to_index,
                                                                     self.rating_scale, prediction_network_params)
        self.prediction_artifacts = prediction_artifacts
        if content_data_used:
            user_vectors = np.concatenate((user_content_vectors, user_vectors), axis=1)
            item_vectors = np.concatenate((item_content_vectors, item_vectors), axis=1)
            assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_output_dims

        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)

        self.log.debug("Fit Method, Before KNN, Unit Length Violations:: user = %s, item = %s",
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))

        _, _ = self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)
        self.fit_done = True
        self.log.info("End Fitting Recommender, user_vectors shape = %s, item_vectors shape = %s, Time to fit = %.1f",
                      user_vectors.shape, item_vectors.shape, time.time() - start_time)
        return user_vectors, item_vectors

    @abc.abstractmethod
    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        pass

    def find_items_for_user(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                            negative: List[Tuple[str, EntityType]] = None) -> List[Tuple[str, float]]:
        results = super().find_items_for_user(user, positive, negative)
        res, dist = zip(*results)
        ratings = self.predict([(user, i) for i in res])
        return list(sorted(zip(res, ratings), key=operator.itemgetter(1), reverse=True))
