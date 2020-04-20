import abc
import operator
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


from .content_recommender import ContentRecommendation
from .logging import getLogger
from .recommendation_base import RecommendationBase
from .utils import unit_length, unit_length_violations


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_collaborative_dims: int = 32):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_dims=n_collaborative_dims)
        self.cb = ContentRecommendation(embedding_mapper, knn_params, rating_scale,
                                        np.inf)
        self.n_collaborative_dims = n_collaborative_dims
        self.content_data_used = None
        self.prediction_artifacts = None
        self.log = getLogger(type(self).__name__)
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
                                                             user_id_to_index: Dict[str, int],
                                                             item_id_to_index: Dict[str, int],
                                                             affinities: List[Tuple[str, str, float]],
                                                             hyperparams: Dict):
        pass

    def __user_item_affinities_triplet_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:

        pass

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
        collaborative_params = {} if "collaborative_params" not in hyperparameters else hyperparameters["collaborative_params"]
        prediction_network_params = {} if "prediction_network_params" not in collaborative_params else \
            collaborative_params["prediction_network_params"]

        use_content = hyperparameters["use_content"] if "use_content" in hyperparameters else False
        content_data_used = ("item_data" in kwargs or "user_data" in kwargs) and use_content
        self.content_data_used = content_data_used

        item_item_affinities: List[Tuple[str, str, bool]] = kwargs[
            "item_item_affinities"] if "item_item_affinities" in kwargs else list()
        user_user_affinities: List[Tuple[str, str, bool]] = kwargs[
            "user_user_affinities"] if "user_user_affinities" in kwargs else list()

        self.log.debug("Hybrid Base: Fit Method: content_data_used = %s", content_data_used)
        start = time.time()
        if content_data_used:
            super(type(self.cb), self.cb).fit(user_ids, item_ids, user_item_affinities, **kwargs)
            user_vectors, item_vectors = self.cb.__build_content_embeddings__(user_ids, item_ids,
                                                                              user_data, item_data,
                                                                              user_item_affinities,
                                                                              np.inf)
            self.cb = None
            del self.cb
            del kwargs["user_data"]
            del kwargs["item_data"]

        else:
            user_vectors, item_vectors = np.random.rand(len(user_ids), 1), np.random.rand(len(item_ids), 1)
        self.log.info("Hybrid Base: Built Content Embedding., user_vectors shape = %s, item vectors shape = %s, Time = %.1f" %
                       (user_vectors.shape, item_vectors.shape, time.time() - start))
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)
        import gc
        gc.collect()

        user_content_vectors, item_content_vectors = user_vectors.copy(), item_vectors.copy()
        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]

        user_vectors, item_vectors = self.__build_collaborative_embeddings__(user_item_affinities,
                                                                             item_item_affinities,
                                                                             user_user_affinities, user_ids, item_ids,
                                                                             user_vectors, item_vectors,
                                                                             collaborative_params)

        self.log.debug("Hybrid Base: Fit Method, Use content = %s, Unit Length Violations:: user_content = %s, item_content = %s" +
                       "user_collab = %s, item_collab = %s", content_data_used,
                       unit_length_violations(user_content_vectors, axis=1), unit_length_violations(item_content_vectors, axis=1),
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)
        gc.collect()

        # assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_collaborative_dims
        self.log.debug("Hybrid Base: Start Building Prediction Network...")
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
        if prediction_artifacts is not None:
            self.prediction_artifacts.update(dict(prediction_artifacts))
        gc.collect()
        self.log.debug("Hybrid Base: Built Prediction Network.")

        self.log.debug("Fit Method, Before KNN, Unit Length Violations:: user = %s, item = %s",
                       unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1))

        user_vectors, item_vectors = self.prepare_for_knn(user_vectors, item_vectors)
        self.knn_user_vectors = user_vectors
        self.knn_item_vectors = item_vectors
        self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)
        self.fit_done = True
        self.log.info("End Fitting Recommender, user_vectors shape = %s, item_vectors shape = %s, Time to fit = %.1f",
                      user_vectors.shape, item_vectors.shape, time.time() - start_time)
        gc.collect()
        return user_vectors, item_vectors

    @abc.abstractmethod
    def prepare_for_knn(self, user_vectors, item_vectors):
        pass

    @abc.abstractmethod
    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        pass

    def find_closest_neighbours(self, user: str, positive: List[Tuple[str, object]] = None,
                                negative: List[Tuple[str, object]] = None, k=None) -> List[Tuple[str, float]]:
        start = time.time()
        results = super().find_closest_neighbours(user, positive, negative, k=k)
        res, dist = zip(*results)
        ratings = self.predict([(user, i) for i in res])
        results = list(sorted(zip(res, ratings), key=operator.itemgetter(1), reverse=True))
        self.log.debug("Find K Items for user = %s, time taken = %.4f",
                      user,
                      time.time() - start)
        return results
