import abc
import operator
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


from .content_recommender import ContentRecommendation
from .logging import getLogger
from .recommendation_base import EntityType
from .recommendation_base import RecommendationBase, FeatureSet
from .utils import unit_length, unit_length_violations


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False, super_fast_inference: bool = False):
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
        self.super_fast_inference = super_fast_inference
        self.prediction_artifacts = dict()

    def __build_collaborative_embeddings__(self, user_item_affinities: List[Tuple[str, str, float]],
                                           item_item_affinities: List[Tuple[str, str, bool]],
                                           user_user_affinities: List[Tuple[str, str, bool]],
                                           user_ids: List[str], item_ids: List[str],
                                           user_vectors: np.ndarray, item_vectors: np.ndarray,
                                           hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:

        self.log.debug("Started Building Collaborative Embeddings...")
        user_item_affinity_fn = self.__user_item_affinities_triplet_trainer__

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

    def __user_item_affinities_triplet_trainer_data_gen_fn__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 hyperparams: Dict):
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        total_users = len(user_ids)
        total_items = len(item_ids)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            affinities = [(user_id_to_index[i], item_id_to_index[j], r) for i, j, r in affinities]

            def get_one_example(i, j, r):
                user = i
                second_item = total_users + j
                distant_item = np.random.randint(0, total_users + total_items)
                return (user, second_item, distant_item), 0

            def generator():
                for i in range(0, len(affinities), batch_size * 4):
                    start = i
                    end = min(i + batch_size * 4, len(affinities))
                    generated = [get_one_example(u, v, w) for u, v, w in affinities[start:end]]
                    for g in generated:
                        yield g

            return generator

        return generate_training_samples

    def __user_item_affinities_triplet_trainer_data_gen__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_item_affinities: List[Tuple[str, str, float]],
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 hyperparams: Dict):

        import tensorflow as tf
        import tensorflow.keras.backend as K
        output_shapes = (((), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64), tf.float32)
        generate_training_samples = self.__user_item_affinities_triplet_trainer_data_gen_fn__(user_ids, item_ids,
                                                                                              user_id_to_index, item_id_to_index,
                                                                                              hyperparams)
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

        import tensorflow as tf
        import tensorflow.keras.backend as K
        from tensorflow import keras
        self.log.debug("Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5
        l2 = hyperparams["l2"] if "l2" in hyperparams else 0.0

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

            # TODO: Unit vectorise here
            vectors = unit_length(vectors, axis=-1)
            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            k = 2
            if l2:
                def embeddings_regularizer(x): return l2 * K.sum(K.relu(K.square(x) - k/embedding_size) + K.relu(1/(k*embedding_size) - K.square(x)))
            else:
                def embeddings_regularizer(x): return 0
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer, embeddings_regularizer=embeddings_regularizer)
            # embeddings_regularizer=embeddings_regularizer
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            # item = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(item)
            item = K.l2_normalize(item, axis=-1)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        bn = build_base_network(n_input_dims, n_output_dims, np.concatenate((user_vectors, item_vectors)))
        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))
        input_3 = keras.Input(shape=(1,))

        item_1 = bn(input_1)
        item_2 = bn(input_2)
        item_3 = bn(input_3)

        i1_i2_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
        i1_i2_dist = 1 - i1_i2_dist

        i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
        i1_i3_dist = 1 - i1_i3_dist

        loss = K.relu(i1_i2_dist - i1_i3_dist + margin)
        model = keras.Model(inputs=[input_1, input_2, input_3],
                            outputs=[loss])

        encoder = bn
        from .tf_utils import LRSchedule
        learning_rate = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        self.log.debug("Started Training User-Item Affinities...")
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
        from surprise import Dataset
        from surprise import Reader
        from surprise import SVDpp, SVD
        reader = Reader(rating_scale=rating_scale)
        train = pd.DataFrame(user_item_affinities)
        train = Dataset.load_from_df(train, reader).build_full_trainset()
        n_epochs = svd_params["n_epochs"] if "n_epochs" in svd_params else 10
        n_factors = svd_params["n_factors"] if "n_factors" in svd_params else 10
        svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs)
        svd_model.fit(train)
        svdpp_model = SVDpp(n_factors=n_factors, n_epochs=n_epochs)
        svdpp_model.fit(train)
        self.prediction_artifacts["svdpp_model"] = svdpp_model
        self.prediction_artifacts["svd_model"] = svd_model

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
        self.log.debug("Hybrid Base: Fit Method Started")
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

        self.log.debug("Hybrid Base: Fit Method: content_data_used = %s, content_dims = %s", content_data_used, self.n_content_dims)
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
        self.prediction_artifacts.update(dict(prediction_artifacts))
        self.__build_svd_model__(user_ids, item_ids, user_item_affinities,
                                             self.user_id_to_index, self.item_id_to_index, self.rating_scale, **svd_params)

        self.log.debug("Fit Method, Before KNN, Unit Length Violations:: user = %s, item = %s",
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))

        user_vectors, item_vectors = self.prepare_for_knn(alpha, content_data_used,
                                                          user_content_vectors, item_content_vectors,
                                                          user_vectors, item_vectors)
        self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)
        self.fit_done = True
        self.log.info("End Fitting Recommender, user_vectors shape = %s, item_vectors shape = %s, Time to fit = %.1f",
                      user_vectors.shape, item_vectors.shape, time.time() - start_time)
        return user_vectors, item_vectors

    @abc.abstractmethod
    def prepare_for_knn(self, alpha, content_data_used,
                        user_content_vectors, item_content_vectors,
                        user_vectors, item_vectors):
        user_content_vectors, item_content_vectors = user_content_vectors * alpha, item_content_vectors * alpha
        user_vectors, item_vectors = user_vectors * (1 - alpha), item_vectors * (1 - alpha)
        if content_data_used:
            user_vectors = np.concatenate((user_content_vectors, user_vectors), axis=1)
            item_vectors = np.concatenate((item_content_vectors, item_vectors), axis=1)
            assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_output_dims
        return user_vectors, item_vectors

    @abc.abstractmethod
    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        pass

    def super_fast_predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        svd_model = self.prediction_artifacts["svd_model"]
        return [svd_model.predict(u, i).est for u, i in user_item_pairs]

    def fast_predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        svdpp_model = self.prediction_artifacts["svdpp_model"]
        return [svdpp_model.predict(u, i).est for u, i in user_item_pairs]

    def find_items_for_user(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                            negative: List[Tuple[str, EntityType]] = None, k=None) -> List[Tuple[str, float]]:
        start = time.time()
        results = super().find_items_for_user(user, positive, negative, k=k)
        res, dist = zip(*results)
        if self.super_fast_inference:
            ratings = self.super_fast_predict([(user, i) for i in res])
        elif self.fast_inference:
            ratings = self.fast_predict([(user, i) for i in res])
        else:
            ratings = self.predict([(user, i) for i in res])
        results = list(sorted(zip(res, ratings), key=operator.itemgetter(1), reverse=True))
        self.log.debug("Find K Items for user = %s, time taken = %.4f",
                      user,
                      time.time() - start)
        return results
