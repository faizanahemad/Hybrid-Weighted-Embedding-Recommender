from .recommendation_base import RecommendationBase, Feature, FeatureSet
from typing import List, Dict, Tuple, Sequence, Type, Set
from sklearn.decomposition import PCA
from scipy.special import comb
from pandas import DataFrame
from .content_embedders import ContentEmbeddingBase
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from joblib import Parallel, delayed
import gc
import sys
import os
from more_itertools import flatten
import dill
from collections import Counter
import operator
from tqdm import tqdm_notebook
import fasttext
from .utils import unit_length


class ContentRecommendation(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: dict, n_output_dims: int = 32,):
        super().__init__(knn_params=knn_params, n_output_dims=n_output_dims)

        self.embedding_mapper: dict[str, ContentEmbeddingBase] = embedding_mapper

    def __build_item_only_embeddings__(self, item_ids: List[str], item_data: FeatureSet):
        item_embeddings = {}
        for feature in item_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            embedding = self.embedding_mapper[feature_name].fit_transform(feature)
            assert embedding.shape[0] == len(item_ids)
            item_embeddings[feature_name] = embedding
        return item_embeddings

    def __build_user_only_embeddings__(self, user_ids: List[str], user_data: FeatureSet):
        user_embeddings = {}
        for feature in user_data:
            feature_name = feature.feature_name
            if feature.feature_type == "id" or feature_name not in self.user_only_features:
                continue
            embedding = self.embedding_mapper[feature_name].fit_transform(feature)
            assert embedding.shape[0] == len(user_ids)
            user_embeddings[feature_name] = embedding
        return user_embeddings

    def __build_item_embeddings__(self, item_ids: List[str],
                                  user_embeddings: Dict[str, np.ndarray],
                                  item_embeddings: Dict[str, np.ndarray],
                                  user_item_affinities: List[Tuple[str, str, float]]):
        item_user_dict: Dict[str, Dict[str, float]] = {}
        for user, item, affinity in user_item_affinities:
            if item not in item_user_dict:
                item_user_dict[item] = {}
            item_user_dict[item][user] = affinity

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
                    item_em = np.average(user_ems, axis=0, weights=weights)
                    item_embedding[i] = item_em
                else:
                    item_embedding[i] = average_embedding.copy()
            item_embeddings[feature_name] = item_embedding
        return item_embeddings

    def __build_user_embeddings__(self,
                                  user_ids: List[str],
                                  user_data: FeatureSet,
                                  item_data: FeatureSet,
                                  user_embeddings: Dict[str, np.ndarray],
                                  item_embeddings: Dict[str, np.ndarray],
                                  user_item_affinities: List[Tuple[str, str, float]]):

        user_item_dict: Dict[str, Dict[str, float]] = {}

        for user, item, affinity in user_item_affinities:
            if user not in user_item_dict:
                user_item_dict[user] = {}
            user_item_dict[user][item] = affinity

        for feature in user_data:
            feature_name = feature.feature_name
            if feature.feature_type == "id" or feature_name in self.user_only_features:
                continue
            embedding = self.embedding_mapper[feature_name].transform(feature)
            user_embeddings[feature_name] = embedding

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
            for i, embedding in enumerate(user_embedding):
                user = user_ids[i]
                if user not in user_item_dict:
                    continue
                item_dict = user_item_dict[user]
                items = item_dict.keys()
                weights = list(item_dict.values())
                item_indices = [self.item_id_to_index[item] for item in items]
                item_ems = np.take(item_embedding, indices=item_indices, axis=0)
                assert len(item_ems) > 0
                item_em = np.average(item_ems, axis=0, weights=weights)
                final_embedding = (embedding + item_em) / 2.0
                user_embedding[i] = final_embedding
            processed_features.append(feature_name)

        # for item_only_features
        for feature in item_data:
            if feature.feature_type == "id" or feature.feature_name in processed_features:
                continue
            feature_name = feature.feature_name
            item_embedding = item_embeddings[feature_name]
            user_embedding = np.zeros(shape=(len(user_ids), item_embedding.shape[1]))
            average_embedding = np.mean(item_embedding, axis=0)
            for i, user in enumerate(user_ids):
                if user in user_item_dict:
                    item_dict = user_item_dict[user]
                    items = item_dict.keys()
                    weights = list(item_dict.values())
                    item_indices = [self.item_id_to_index[item] for item in items]
                    item_ems = np.take(item_embedding, indices=item_indices, axis=0)
                    assert len(item_ems) > 0
                    item_em = np.average(item_ems, axis=0, weights=weights)
                    user_embedding[i] = item_em
                else:
                    user_embedding[i] = average_embedding.copy()
            user_embeddings[feature_name] = user_embedding
            processed_features.append(feature_name)
        return user_embeddings, processed_features

    def __concat_feature_vectors__(self, processed_features, item_embeddings, user_embeddings):
        # Unit Vectorise Features
        for feature_name in processed_features:
            item_embedding = unit_length(item_embeddings[feature_name], axis=1)
            user_embedding = unit_length(user_embeddings[feature_name], axis=1)
            item_embeddings[feature_name] = item_embedding
            user_embeddings[feature_name] = user_embedding
        # Concat Features

        user_vectors = user_embeddings[processed_features[0]]
        item_vectors = item_embeddings[processed_features[0]]

        for feature_name in processed_features[1:]:
            user_vectors = np.concatenate((user_vectors, user_embeddings[feature_name]), axis=1)
            item_vectors = np.concatenate((item_vectors, item_embeddings[feature_name]), axis=1)

        # PCA
        user_vectors_length = len(user_vectors)
        all_vectors = np.concatenate((user_vectors, item_vectors), axis=0)
        if self.n_output_dims < all_vectors.shape[1]:
            all_vectors = PCA(n_components=self.n_output_dims, ).fit_transform(all_vectors)
        user_vectors = all_vectors[:user_vectors_length]
        item_vectors = all_vectors[user_vectors_length:]

        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)
        return user_vectors,item_vectors

    def __build_content_embeddings__(self,
                                     user_ids: List[str],
                                     item_ids: List[str],
                                     user_data: FeatureSet,
                                     item_data: FeatureSet,
                                     user_item_affinities: List[Tuple[str, str, float]]):
        item_embeddings = self.__build_item_only_embeddings__(item_ids, item_data)
        user_embeddings = self.__build_user_only_embeddings__(user_ids, user_data)
        item_embeddings = self.__build_item_embeddings__(item_ids, user_embeddings,
                                                         item_embeddings, user_item_affinities)
        user_embeddings, processed_features = self.__build_user_embeddings__(user_ids, user_data, item_data,
                                                                             user_embeddings, item_embeddings,
                                                                             user_item_affinities)

        user_vectors, item_vectors = self.__concat_feature_vectors__(processed_features, item_embeddings,
                                                                     user_embeddings)
        return user_vectors, item_vectors

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            **kwargs):
        """
        Note: Users Features need to be a subset of item features for them to be embedded into same space

        Currently Only supports text data, Support for Categorical and numerical data to be added using AutoEncoders
        :param user_ids:
        :param item_ids:
        :param warm_start:
        :param kwargs:
        :return:
        """

        super().fit(user_ids, item_ids, **kwargs)
        item_data: FeatureSet = kwargs["item_data"]
        user_data: FeatureSet = kwargs["user_data"]
        user_item_affinities: List[Tuple[str, str, float]] = kwargs["user_item_affinities"]

        user_vectors, item_vectors = self.__build_content_embeddings__(user_ids, item_ids,
                                                                       user_data, item_data, user_item_affinities)

        _, _ = self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)

        # AutoEncoder them so that error is minimised and distance is maintained
        # https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances
        # Distance Preserving vs Non Preserving

        self.fit_done = True
        return user_vectors, item_vectors

    def default_predictions(self):
        assert self.fit_done
        raise NotImplementedError()

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
