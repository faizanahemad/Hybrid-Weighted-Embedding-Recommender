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
                 n_output_dims: int = 32):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_output_dims=n_output_dims)

        self.embedding_mapper: dict[str, ContentEmbeddingBase] = embedding_mapper
        self.log = getLogger(type(self).__name__)

    def __build_user_only_embeddings__(self, user_ids: List[str], user_data: FeatureSet):
        self.log.debug("ContentRecommendation::__build_user_only_embeddings__:: Building User Only Embedding ...")
        user_embeddings = {}
        for feature in user_data:
            feature_name = feature.feature_name
            if feature.feature_type != "id" and feature_name in self.user_only_features:
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

        item_user_dict: Dict[str, Dict[str, float]] = build_item_user_dict(user_item_affinities)
        for feature_name in self.user_only_features:
            user_embedding = user_embeddings[feature_name]
            item_embedding = np.zeros(shape=(len(item_ids), user_embedding.shape[1]))
            average_embedding = np.mean(user_embedding, axis=0)
            for i, item in enumerate(item_ids):
                if item in item_user_dict:
                    user_dict = item_user_dict[item]
                    users = user_dict.keys()
                    weights = list(user_dict.values())
                    user_indices = [self.user_id_to_index[user] for user in users]
                    user_ems = np.take(user_embedding, indices=user_indices, axis=0)
                    assert len(user_ems) > 0
                    weights[weights == 0] = 1e-3
                    item_em = np.average(user_ems, axis=0, weights=weights)
                    item_embedding[i] = item_em
                else:
                    item_embedding[i] = average_embedding.copy()
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
        user_item_dict: Dict[str, Dict[str, float]] = build_user_item_dict(user_item_affinities)

        user_item_idx_list: Dict[str, List[int]] = {}
        for user, item, affinity in user_item_affinities:
            if user not in user_item_idx_list:
                user_item_idx_list[user] = [self.item_id_to_index[item]]
            else:
                user_item_idx_list[user].append(self.item_id_to_index[item])

        for feature in user_data:
            feature_name = feature.feature_name
            if feature.feature_type != "id" and feature_name not in self.user_only_features:
                embedding = self.embedding_mapper[feature_name].transform(feature)
                if np.sum(np.isnan(embedding)) != 0:
                    self.log.info("User Embedding for Item Feature: Feature = %s, Nan Users = %s", feature_name, get_nan_rows(embedding))
                assert np.sum(np.isnan(embedding)) == 0
                embedding = unit_length(embedding, axis=1)
                user_embeddings[feature_name] = embedding
        self.log.debug("ContentRecommendation::__build_user_embeddings__:: User Embeddings Transform Done for User+Item Combo features")

        self.log.debug("ContentRecommendation::__build_user_embeddings__:: Start Processing User Embeddings as average of Item Embeddings")

        # For features which are not in user_data take average of item_features, while for ones present follow above method
        # Assume some features are not present in Users
        # Weighted Averaging for features not present and present in user_data

        # for features which are in both user_data and item_data or for user_only_features too
        processed_features = []
        for feature in user_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            user_embedding = user_embeddings[feature_name]
            item_embedding = item_embeddings[feature_name]
            user_embedding = unit_length(user_embedding, axis=1)
            item_embedding = unit_length(item_embedding, axis=1)
            self.log.debug("ContentRecommendation::__build_user_embeddings__:: Processing User+Item Embeddings for feature = %s" % (feature_name))

            if np.sum(np.isnan(user_embedding)) != 0:
                self.log.info("User Embedding: Feature = %s, Nan Users = %s", feature_name,
                              get_nan_rows(user_embedding))
            assert np.sum(np.isnan(item_embedding)) == 0
            if np.sum(np.isnan(item_embedding)) != 0:
                self.log.info("Item Embedding: Feature = %s, Nan Users = %s", feature_name,
                              get_nan_rows(item_embedding))
            assert np.sum(np.isnan(user_embedding)) == 0
            for i, (user, embedding) in enumerate(zip(user_ids, user_embedding)):
                if user not in user_item_idx_list:
                    continue
                item_indices = user_item_idx_list[user]
                item_em = item_embedding[item_indices].mean(0)
                final_embedding = (embedding + item_em) / 2.0
                user_embedding[i] = final_embedding
                if i % int(len(user_embedding)/20) == 0:
                    self.log.debug("ContentRecommendation::__build_user_embeddings__:: Processed %s/%s User+Item Embeddings of feature %s"
                                   % (i, len(user_embedding), feature_name))
            user_embedding = unit_length(user_embedding, axis=1)
            user_embeddings[feature_name] = user_embedding
            processed_features.append(feature_name)

        # for item_only_features
        for feature in item_data:
            if feature.feature_type == "id" or feature.feature_name in processed_features:
                continue
            feature_name = feature.feature_name
            item_embedding = item_embeddings[feature_name]
            item_embedding = unit_length(item_embedding, axis=1)
            user_embedding = np.ones(shape=(len(user_ids), item_embedding.shape[1]))
            self.log.debug(
                "ContentRecommendation::__build_user_embeddings__:: Processing Item Only Embeddings for feature = %s" % (
                    feature_name))
            for i, user in enumerate(user_ids):
                if user in user_item_idx_list:
                    item_indices = user_item_idx_list[user]
                    item_em = item_embedding[item_indices].mean(0)
                    user_embedding[i] = item_em
                if i % int(len(user_embedding)/20) == 0:
                    self.log.debug("ContentRecommendation::__build_user_embeddings__:: Processed %s/%s Item Only Embeddings of feature %s"
                                   % (i, len(user_embedding), feature_name))
            user_embedding = unit_length(user_embedding, axis=1)
            user_embeddings[feature_name] = user_embedding
            processed_features.append(feature_name)
        self.log.debug(
            "ContentRecommendation::__build_user_embeddings__:: Built User Embedding for %s, Processed features = %s" % (list(
                user_embeddings.keys()), processed_features))
        return user_embeddings, processed_features

    def __concat_feature_vectors__(self, processed_features, item_embeddings, user_embeddings, n_output_dims):
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
        if n_output_dims < all_vectors.shape[1]:
            # all_vectors = StandardScaler().fit_transform(all_vectors)
            pca = PCA(n_components=n_output_dims, )
            all_vectors = pca.fit_transform(all_vectors)
            all_vectors = StandardScaler().fit_transform(all_vectors)
            self.log.info("Content Recommender::__concat_feature_vectors__, PCA explained variance:  %.4f, explained variance ratio: %.4f",
                           np.sum(pca.explained_variance_), np.sum(pca.explained_variance_ratio_))

        if n_output_dims > all_vectors.shape[1]:
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
                                     n_output_dims):

        user_embeddings = self.__build_user_only_embeddings__(user_ids, user_data)
        item_embeddings = self.__build_item_embeddings__(item_ids, user_embeddings,
                                                         item_data, user_item_affinities)
        user_embeddings, processed_features = self.__build_user_embeddings__(user_ids, user_data, item_data,
                                                                             user_embeddings, item_embeddings,
                                                                             user_item_affinities)

        self.log.info("Concatenating Content Embeddings ... ")
        user_vectors, item_vectors = self.__concat_feature_vectors__(processed_features, item_embeddings,
                                                                     user_embeddings, n_output_dims)
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
                                                                       self.n_output_dims)

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
