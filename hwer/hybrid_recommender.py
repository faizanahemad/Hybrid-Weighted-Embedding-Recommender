import abc
import operator
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
import pandas as pd

from .content_recommender import ContentRecommendation
from .logging import getLogger
from .recommendation_base import EntityType
from .recommendation_base import RecommendationBase, FeatureSet
from .utils import unit_length, normalize_affinity_scores_by_user, UnitLengthRegularization, unit_length_violations, \
    LRSchedule, \
    resnet_layer_with_content


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_output_dims=n_content_dims + n_collaborative_dims)
        self.cb = ContentRecommendation(embedding_mapper, knn_params, rating_scale,
                                        n_content_dims, )
        self.n_content_dims = n_content_dims
        self.n_collaborative_dims = n_collaborative_dims
        self.content_data_used = None
        self.prediction_artifacts = None
        self.log = getLogger(type(self).__name__)
        self.fast_inference = fast_inference

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
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.2
        random_positive_weight = hyperparams[
            "random_positive_weight"] if "random_positive_weight" in hyperparams else 0.1
        random_negative_weight = hyperparams[
            "random_negative_weight"] if "random_negative_weight" in hyperparams else 0.25
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.1

        total_items = len(entity_ids)
        aff_range = np.max([r for u1, u2, r in entity_entity_affinities]) - np.min([r for u1, u2, r in entity_entity_affinities])
        random_positive_weight = random_positive_weight * aff_range
        random_negative_weight = random_negative_weight * aff_range

        assert np.sum(np.isnan(vectors)) == 0

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            item_close_dict = defaultdict(list)
            item_far_dict = defaultdict(list)
            affinities = [(entity_id_to_index[i], entity_id_to_index[j], r) for i, j, r in affinities]
            for i, j, r in affinities:
                if r > 0:
                    item_close_dict[i].append((j, r))
                    item_close_dict[j].append((i, r))

                if r <= 0:
                    item_far_dict[i].append((j, r))
                    item_far_dict[j].append((i, r))

            def triplet_wt_fn(x): return 1 + np.log1p(np.abs(x / aff_range))

            def get_one_example(i, j, r):
                first_item = i
                second_item = j
                random_item = np.random.randint(0, total_items)
                choose_random_pair = np.random.rand() < (random_pair_proba if r > 0 else random_pair_proba / 100)
                if r < 0:
                    distant_item = second_item
                    distant_item_weight = r

                    if choose_random_pair or i not in item_close_dict:
                        second_item, close_item_weight = random_item, random_positive_weight

                    else:
                        second_item, close_item_weight = item_close_dict[i][
                            np.random.randint(0, len(item_close_dict[i]))]
                else:
                    close_item_weight = r
                    if choose_random_pair or i not in item_far_dict:
                        distant_item, distant_item_weight = random_item, random_negative_weight

                    else:
                        distant_item, distant_item_weight = item_far_dict[i][
                            np.random.randint(0, len(item_far_dict[i]))]

                close_item_weight = triplet_wt_fn(close_item_weight)
                distant_item_weight = triplet_wt_fn(distant_item_weight)
                return (first_item, second_item, distant_item, close_item_weight, distant_item_weight), 0

            def generator():
                for i in range(0, len(affinities), batch_size*4):
                    start = i
                    end = min(i + batch_size*4, len(affinities))
                    generated = [get_one_example(u, v, w) for u, v, w in affinities[start:end]]
                    for g in generated:
                        yield g
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
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(entity_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(embedding_size, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
            item = dense(item)
            item = UnitLengthRegularization(l1=0.1)(item)
            # item = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(item)
            item = K.l2_normalize(item, axis=-1)
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
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0
        return user_vectors, item_vectors



    def __user_item_affinities_triplet_trainer_data_gen__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_item_affinities: List[Tuple[str, str, float]],
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 hyperparams: Dict):
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        random_pair_user_item_proba = hyperparams[
            "random_pair_user_item_proba"] if "random_pair_user_item_proba" in hyperparams else 0.2
        total_users = len(user_ids)
        total_items = len(item_ids)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            affinities = [(user_id_to_index[i], item_id_to_index[j], r) for i, j, r in affinities]

            def get_one_example(i, j, r):
                user = i
                second_item = total_users + j
                random_item = total_users + np.random.randint(0, total_items)
                random_user = np.random.randint(0, total_users)
                choose_user_pair = np.random.rand() < random_pair_user_item_proba
                close_item_weight = 1
                distant_item, distant_item_weight = random_user if choose_user_pair else random_item, 1
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

        return train

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
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        n_input_dims = user_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        train = self.__user_item_affinities_triplet_trainer_data_gen__(user_ids, item_ids,
                                                 user_item_affinities,
                                                 user_id_to_index, item_id_to_index,
                                                 hyperparams)
        train = train.shuffle(batch_size * 10).batch(batch_size).prefetch(32)

        total_users = len(user_ids)

        def build_base_network(embedding_size, n_output_dims, vectors):
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            dense = keras.layers.Dense(n_output_dims, activation="tanh", use_bias=False, kernel_initializer="glorot_uniform")
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
        i1_i2_dist = i1_i2_dist

        i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
        i1_i3_dist = 1 - i1_i3_dist
        i1_i3_dist = i1_i3_dist

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

    def __build_svd_model__(self, user_ids: List[str], item_ids: List[str],
                                 user_item_affinities: List[Tuple[str, str, float]],
                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                 rating_scale: Tuple[float, float], **svd_params):
        reader = Reader(rating_scale=rating_scale)
        train = pd.DataFrame(user_item_affinities)
        train = Dataset.load_from_df(train, reader).build_full_trainset()
        n_epochs = svd_params["n_epochs"] if "n_epochs" in svd_params else 10
        n_factors = svd_params["n_factors"] if "n_factors" in svd_params else 10
        svd_model = SVDpp(n_factors=n_factors, n_epochs=n_epochs)
        svd_model.fit(train)
        return svd_model

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
        svd_params = {} if "svd_params" not in hyperparameters else hyperparameters["svd_params"]
        collaborative_params = {} if "collaborative_params" not in hyperparameters else hyperparameters["collaborative_params"]
        prediction_network_params = {} if "prediction_network_params" not in collaborative_params else \
            collaborative_params["prediction_network_params"]

        combining_factor: int = hyperparameters["combining_factor"] if "combining_factor" in hyperparameters else 0.5
        alpha = combining_factor
        assert 0 <= alpha <= 1
        use_content = prediction_network_params["use_content"] if "use_content" in prediction_network_params else False
        content_data_used = ("item_data" in kwargs or "user_data" in kwargs) and alpha > 0 and use_content
        self.content_data_used = content_data_used

        self.n_output_dims = self.n_content_dims + self.n_collaborative_dims if content_data_used else self.n_collaborative_dims

        item_item_affinities: List[Tuple[str, str, bool]] = kwargs[
            "item_item_affinities"] if "item_item_affinities" in kwargs else list()
        user_user_affinities: List[Tuple[str, str, bool]] = kwargs[
            "user_user_affinities"] if "user_user_affinities" in kwargs else list()

        self.log.debug("Fit Method: content_data_used = %s, content_dims = %s", content_data_used, self.n_content_dims)
        if content_data_used:
            super(type(self.cb), self.cb).fit(user_ids, item_ids, user_item_affinities, **kwargs)
            user_vectors, item_vectors = self.cb.__build_content_embeddings__(user_ids, item_ids,
                                                                              user_data, item_data,
                                                                              user_item_affinities,
                                                                              self.n_content_dims)
        else:
            user_vectors, item_vectors = np.random.rand(len(user_ids), self.n_content_dims), np.random.rand(len(item_ids), self.n_content_dims)
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)

        user_content_vectors, item_content_vectors = user_vectors.copy(), item_vectors.copy()
        assert user_content_vectors.shape[1] == item_content_vectors.shape[1] == self.n_content_dims

        user_vectors, item_vectors = self.__build_collaborative_embeddings__(user_item_affinities,
                                                                             item_item_affinities,
                                                                             user_user_affinities, user_ids, item_ids,
                                                                             user_vectors, item_vectors,
                                                                             collaborative_params)

        self.log.debug("Fit Method, Use content = %s, Unit Length Violations:: user_content = %s, item_content = %s" +
                       "user_collab = %s, item_collab = %s", content_data_used,
                       unit_length_violations(user_content_vectors, axis=1), unit_length_violations(item_content_vectors, axis=1),
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)

        assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_collaborative_dims
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
        svd_model = self.__build_svd_model__(user_ids, item_ids, user_item_affinities,
                                             self.user_id_to_index, self.item_id_to_index, self.rating_scale, **svd_params)
        self.prediction_artifacts["svd_model"] = svd_model

        user_content_vectors, item_content_vectors = user_content_vectors * alpha, item_content_vectors * alpha
        user_vectors, item_vectors = user_vectors * (1 - alpha), item_vectors * (1 - alpha)
        if content_data_used:
            user_vectors = np.concatenate((user_content_vectors, user_vectors), axis=1)
            item_vectors = np.concatenate((item_content_vectors, item_vectors), axis=1)
            assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_output_dims

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

    def fast_predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        svd_model = self.prediction_artifacts["svd_model"]
        return [svd_model.predict(u, i).est for u, i in user_item_pairs]

    def find_items_for_user(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                            negative: List[Tuple[str, EntityType]] = None, k=None) -> List[Tuple[str, float]]:
        start = time.time()
        results = super().find_items_for_user(user, positive, negative, k=k)
        res, dist = zip(*results)
        if self.fast_inference:
            ratings = self.fast_predict([(user, i) for i in res])
        else:
            ratings = self.predict([(user, i) for i in res])
        results = list(sorted(zip(res, ratings), key=operator.itemgetter(1), reverse=True))
        self.log.info("Find K Items for user = %s, time taken = %.4f",
                      user,
                      time.time() - start)
        return results
