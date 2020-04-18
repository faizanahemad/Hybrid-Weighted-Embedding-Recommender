from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .content_embedders import ContentEmbeddingBase
from .logging import getLogger
from .recommendation_base import RecommendationBase, FeatureSet
from .utils import unit_length, build_user_item_dict, build_item_user_dict, get_nan_rows


class ContentRecommendation(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_dims: int = 32):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_dims=n_dims)

        self.embedding_mapper: dict[str, ContentEmbeddingBase] = embedding_mapper
        self.log = getLogger(type(self).__name__)

    def __build_user_only_embeddings__(self, user_ids: List[str], user_data: FeatureSet):
        self.log.debug("ContentRecommendation::__build_user_only_embeddings__:: Building User Only Embedding ...")
        user_embeddings = {}
        for feature in user_data:
            feature_name = feature.feature_name
            if feature.feature_type != "id":
                embedding = self.embedding_mapper[feature_name].fit_transform(feature)
                assert embedding.shape[0] == len(user_ids)
                if np.sum(np.isnan(embedding)) != 0:
                    self.log.info("User Only Embedding: Feature = %s, Nan Users = %s", feature_name, get_nan_rows(embedding))
                assert np.sum(np.isnan(embedding)) == 0
                embedding = unit_length(embedding, axis=1)
                user_embeddings[feature_name] = embedding
                self.log.debug("ContentRecommendation::__build_user_only_embeddings__:: Finished feature = %s for User only Embedding" % feature_name)

        self.log.debug("ContentRecommendation::__build_user_only_embeddings__:: Built User Only Embedding for %s" % list(user_embeddings.keys()))
        return user_embeddings

    def __build_item_embeddings__(self, item_ids: List[str],
                                  user_embeddings: Dict[str, np.ndarray],
                                  item_data: FeatureSet,
                                  user_item_affinities: List[Tuple[str, str, float]]):
        self.log.debug("ContentRecommendation::__build_item_embeddings__:: Building Item Embedding ...")
        item_embeddings = {}
        for feature in item_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            embedding = self.embedding_mapper[feature_name].fit_transform(feature)
            assert embedding.shape[0] == len(item_ids)
            if np.sum(np.isnan(embedding)) != 0:
                self.log.info("Item Embedding: Feature = %s, Nan Items = %s", feature_name,
                              get_nan_rows(embedding))
            assert np.sum(np.isnan(embedding)) == 0
            embedding = unit_length(embedding, axis=1)
            item_embeddings[feature_name] = embedding

        item_user_idx_list: Dict[str, List[int]] = {}
        for user, item, affinity in user_item_affinities:
            if item not in item_user_idx_list:
                item_user_idx_list[item] = [self.user_id_to_index[user]]
            else:
                item_user_idx_list[item].append(self.user_id_to_index[user])
        for feature_name in self.user_only_features:
            user_embedding = user_embeddings[feature_name]
            item_embedding = np.full(shape=(len(item_ids), user_embedding.shape[1]), fill_value=1/np.sqrt(user_embedding.shape[1]))
            for i, item in enumerate(item_ids):
                if item not in item_user_idx_list:
                    continue
                user_indices = item_user_idx_list[item]
                item_embedding[i] = user_embedding[user_indices].mean(0)
            if np.sum(np.isnan(item_embedding)) != 0:
                self.log.info("Item Embedding:user_only_features: Feature = %s, Nan Items = %s", feature_name,
                              get_nan_rows(item_embedding))
            assert np.sum(np.isnan(item_embedding)) == 0
            item_embedding = unit_length(item_embedding, axis=1)
            item_embeddings[feature_name] = item_embedding
        self.log.debug(
            "ContentRecommendation::__build_item_embeddings__:: Built Item Embedding for %s" % list(
                item_embeddings.keys()))
        return item_embeddings

    def __build_user_embeddings__(self,
                                  user_ids: List[str],
                                  user_data: FeatureSet,
                                  item_data: FeatureSet,
                                  user_embeddings: Dict[str, np.ndarray],
                                  item_embeddings: Dict[str, np.ndarray],
                                  user_item_affinities: List[Tuple[str, str, float]]):

        self.log.debug("ContentRecommendation::__build_user_embeddings__:: Building User Embedding ...")
        user_item_idx_list: Dict[str, List[int]] = {}
        for user, item, affinity in user_item_affinities:
            if user not in user_item_idx_list:
                user_item_idx_list[user] = [self.item_id_to_index[item]]
            else:
                user_item_idx_list[user].append(self.item_id_to_index[item])

        # for item_only_features
        for feature in item_data:
            if feature.feature_type == "id" or feature.feature_name in user_data.feature_names:
                continue
            feature_name = feature.feature_name
            item_embedding = item_embeddings[feature_name]
            item_embedding = unit_length(item_embedding, axis=1)
            user_embedding = np.ones(shape=(len(user_ids), item_embedding.shape[1]))
            self.log.debug(
                "ContentRecommendation::__build_user_embeddings__:: Processing Item Only Embeddings for feature = %s" % (
                    feature_name))
            for i, user in enumerate(user_ids):
                if user not in user_item_idx_list:
                    continue
                item_indices = user_item_idx_list[user]
                item_em = item_embedding[item_indices].mean(0)
                user_embedding[i] = item_em
                if i % int(len(user_embedding)/20) == 0:
                    self.log.debug("ContentRecommendation::__build_user_embeddings__:: Processed %s/%s Item Only Embeddings of feature %s"
                                   % (i, len(user_embedding), feature_name))
            user_embedding = unit_length(user_embedding, axis=1)
            user_embeddings[feature_name] = user_embedding
        self.log.debug(
            "ContentRecommendation::__build_user_embeddings__:: Built User Embedding for %s" % (list(
                user_embeddings.keys())))
        return user_embeddings

    def __concat_feature_vectors__(self, item_embeddings, user_embeddings, n_dims):
        processed_features = list(item_embeddings.keys()) + list(user_embeddings.keys())
        self.log.debug("ContentRecommendation::__concat_feature_vectors__:: Concat Features = %s, " % processed_features)
        # Making Each Embedding vector to have unit length Features
        for feature_name in processed_features:
            item_embedding = unit_length(item_embeddings[feature_name], axis=1)
            user_embedding = unit_length(user_embeddings[feature_name], axis=1)
            item_embeddings[feature_name] = item_embedding
            user_embeddings[feature_name] = user_embedding
        # Concat Features

        user_vectors = user_embeddings[processed_features[0]]
        item_vectors = item_embeddings[processed_features[0]]
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        for feature_name in processed_features[1:]:
            user_vectors = np.concatenate((user_vectors, user_embeddings[feature_name]), axis=1)
            item_vectors = np.concatenate((item_vectors, item_embeddings[feature_name]), axis=1)
            if np.sum(np.isnan(user_vectors)) != 0 or np.sum(np.isnan(item_vectors)) != 0:
                self.log.info("Feature = %s, Nan Users = %s, Nan Items = %s", feature_name, get_nan_rows(user_vectors), get_nan_rows(item_vectors))

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0
        # PCA
        user_vectors_length = len(user_vectors)
        all_vectors = np.concatenate((user_vectors, item_vectors), axis=0)
        if n_dims < all_vectors.shape[1]:
            # all_vectors = StandardScaler().fit_transform(all_vectors)
            pca = PCA(n_components=n_dims, )
            all_vectors = pca.fit_transform(all_vectors)
            all_vectors = StandardScaler().fit_transform(all_vectors)
            self.log.info("Content Recommender::__concat_feature_vectors__, PCA explained variance:  %.4f, explained variance ratio: %.4f",
                           np.sum(pca.explained_variance_), np.sum(pca.explained_variance_ratio_))

        if n_dims > all_vectors.shape[1] and n_dims != np.inf:
            raise AssertionError("Output Dims are higher than Total Feature Dims.")
        all_vectors = unit_length(all_vectors, axis=1)
        user_vectors = all_vectors[:user_vectors_length]
        item_vectors = all_vectors[user_vectors_length:]
        self.log.debug(
            "ContentRecommendation::__concat_feature_vectors__:: Concat Features Done, user_vectors = %s, item_vectors = %s" % (user_vectors.shape, item_vectors.shape))
        return user_vectors, item_vectors

    def __build_content_embeddings__(self,
                                     user_ids: List[str],
                                     item_ids: List[str],
                                     user_data: FeatureSet,
                                     item_data: FeatureSet,
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     n_dims):

        user_embeddings = self.__build_user_only_embeddings__(user_ids, user_data)
        item_embeddings = self.__build_item_embeddings__(item_ids, user_embeddings,
                                                         item_data, user_item_affinities)
        user_embeddings = self.__build_user_embeddings__(user_ids, user_data, item_data,
                                                                             user_embeddings, item_embeddings,
                                                                             user_item_affinities)

        self.log.info("Concatenating Content Embeddings ... ")
        user_vectors, item_vectors = self.__concat_feature_vectors__(item_embeddings,
                                                                     user_embeddings, n_dims)
        self.log.info("Built Content Embeddings, user vectors shape = %s, item vectors shape = %s", user_vectors.shape, item_vectors.shape)
        return user_vectors, item_vectors

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_item_affinities: List[Tuple[str, str, float]],
            **kwargs):
        """

        :param user_ids:
        :param item_ids:
        :param user_item_affinities:
        :param kwargs:
        :return:
        """

        user_item_affinities = super().fit(user_ids, item_ids, user_item_affinities, **kwargs)
        item_data: FeatureSet = kwargs["item_data"] if "item_data" in kwargs else FeatureSet([])
        user_data: FeatureSet = kwargs["user_data"] if "user_data" in kwargs else FeatureSet([])

        user_vectors, item_vectors = self.__build_content_embeddings__(user_ids, item_ids,
                                                                       user_data, item_data, user_item_affinities,
                                                                       self.n_dims)

        self.knn_user_vectors = user_vectors
        self.knn_item_vectors = item_vectors
        self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)

        # AutoEncoder them so that error is minimised and distance is maintained
        # https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances
        # Distance Preserving vs Non Preserving

        self.fit_done = True
        return user_vectors, item_vectors

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        return [self.mu + self.bu[u] + self.bi[i] for u, i in user_item_pairs]

    def default_predictions(self):
        assert self.fit_done
        raise NotImplementedError()

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
