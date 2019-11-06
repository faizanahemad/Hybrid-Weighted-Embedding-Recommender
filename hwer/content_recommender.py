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
import nmslib


class ContentRecommendation(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: dict, n_output_dims: int = 32,):
        super().__init__()
        self.n_output_dims = n_output_dims
        self.embedding_mapper: dict[str, ContentEmbeddingBase] = embedding_mapper
        self.knn_params = knn_params
        if self.knn_params is None:
            self.knn_params = dict(n_neighbors=1000,
                                index_time_params = {'M': 15, 'indexThreadQty': 16, 'efConstruction': 200, 'post': 0, 'delaunay_type': 1})

        self.user_id_to_vector = None
        self.index_to_user_id = None
        self.item_id_to_vector = None
        self.index_to_item_id = None
        self.user_knn = None
        self.item_knn = None

        self.fit_done = False

    def add_users(self, users: List[str]):
        super().add_users(users)

    def add_items(self, items: List[str]):
        super().add_items(items)

    def __build_knn__(self, user_vectors: np.ndarray, item_vectors: np.ndarray):
        n_neighbors = self.knn_params["n_neighbors"]
        index_time_params = self.knn_params["index_time_params"]
        query_time_params = {'efSearch': n_neighbors}


        nms_user_index = nmslib.init(method='hnsw', space='cosinesimil')
        nms_user_index.addDataPointBatch(user_vectors)
        nms_user_index.createIndex(index_time_params, print_progress=True)
        nms_user_index.setQueryTimeParams(query_time_params)

        nms_item_index = nmslib.init(method='hnsw', space='cosinesimil')
        nms_item_index.addDataPointBatch(item_vectors)
        nms_item_index.createIndex(index_time_params, print_progress=True)
        nms_item_index.setQueryTimeParams(query_time_params)

        return nms_user_index, nms_item_index

    def __build_user_embeddings__(self,
                                  user_ids: List[str],
                                  item_ids: List[str],
                                  item_embeddings: Dict[str, np.ndarray], **kwargs):

        item_data: FeatureSet = kwargs["item_data"]
        user_data: FeatureSet = kwargs["user_data"]
        user_item_affinities: List[Tuple[str, str, float]] = kwargs["user_item_affinities"]

        user_item_dict: Dict[str, Dict[str, float]] = {}

        item_to_index = dict(zip(item_ids, range(len(item_ids))))

        for user, item, affinity in user_item_affinities:
            if user not in user_item_dict:
                user_item_dict[user] = {}
            user_item_dict[user][item] = affinity

        user_embeddings = {}
        for feature in user_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            embedding = self.embedding_mapper[feature_name].predict(feature)
            user_embeddings[feature_name] = embedding

        # For features which are not in user_data take average of item_features, while for ones present follow above method
        # Assume some features are not present in Users
        # Weighted Averaging for features not present and present in user_data

        processed_features = []
        for feature in user_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            user_embedding = user_embeddings[feature_name]
            item_embedding = item_embeddings[feature_name]
            for i, embedding in enumerate(user_embedding):
                user = user_ids[i]
                item_dict = user_item_dict[user]
                items = item_dict.keys()
                weights = item_dict.values()
                item_indices = [item_to_index[item] for item in items]
                item_ems = np.take(item_embedding, indices=item_indices, axis=0)
                assert len(item_ems) > 0
                item_em = np.average(item_ems, axis=0, weights=weights)
                final_embedding = (embedding + item_em) / 2.0
                user_embedding[i] = final_embedding
            processed_features.append(feature_name)

        for feature in item_data:
            if feature.feature_type == "id" or feature.feature_name in processed_features:
                continue
            feature_name = feature.feature_name
            item_embedding = item_embeddings[feature_name]
            user_embedding = np.zeros(shape=(len(user_ids), item_embedding.shape[1]))
            for i, user in user_ids:
                item_dict = user_item_dict[user]
                items = item_dict.keys()
                weights = item_dict.values()
                item_indices = [item_to_index[item] for item in items]
                item_ems = np.take(item_embedding, indices=item_indices, axis=0)
                assert len(item_ems) > 0
                item_em = np.average(item_ems, axis=0, weights=weights)
                user_embedding[i] = item_em
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
                                     **kwargs):
        item_data: FeatureSet = kwargs["item_data"]

        item_embeddings = {}
        for feature in item_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            embedding = self.embedding_mapper[feature_name].fit_transform(feature)
            item_embeddings[feature_name] = embedding

        # Make one ndarray
        # Make user_averaged_item_vectors

        user_embeddings, processed_features = self.__build_user_embeddings__(user_ids, item_ids,
                                                                             item_embeddings, **kwargs)
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
        user_vectors, item_vectors = self.__build_content_embeddings__(user_ids, item_ids, **kwargs)

        self.user_id_to_vector = dict(zip(user_ids, user_vectors))
        self.index_to_user_id = dict(zip(range(len(user_ids)), user_ids))

        self.item_id_to_vector = dict(zip(item_ids, user_vectors))
        self.index_to_item_id = dict(zip(range(len(item_ids)), item_ids))

        nms_user_index, nms_item_index = self.__build_knn__(user_vectors, item_vectors)
        self.user_knn = nms_user_index
        self.item_knn = nms_item_index

        # AutoEncoder them so that error is minimised and distance is maintained
        # https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances
        # Distance Preserving vs Non Preserving

        self.fit_done = True
        return user_vectors, item_vectors

    def get_average_embeddings(self, entities: List[str]):
        embeddings = []
        for entity in entities:
            if entity in self.user_id_to_vector:
                embeddings.append(self.user_id_to_vector[entity])
            elif entity in self.item_id_to_vector:
                embeddings.append(self.item_id_to_vector[entity])
            else:
                raise ValueError("Unseen entity: %s"%(entity))
        return np.average(embeddings, axis=0)

    def default_predictions(self):
        assert self.fit_done
        raise NotImplementedError()

    def find_similar_items(self, item: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert item in self.item_id_to_vector
        embedding_list = [self.item_id_to_vector[item], self.get_average_embeddings(positive), -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.item_knn.knnQuery(embedding)
        return [(self.index_to_item_id[idx], dt) for idx, dt in zip(neighbors,dist)]

    def find_similar_users(self, user: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert user in self.user_id_to_vector
        embedding_list = [self.user_id_to_vector[user], self.get_average_embeddings(positive),
                          -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.user_knn.knnQuery(embedding)
        return [(self.index_to_user_id[idx], dt) for idx, dt in zip(neighbors,dist)]

    def find_items_for_user(self, user: List[str], positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert user in self.user_id_to_vector
        embedding_list = [self.user_id_to_vector[user], self.get_average_embeddings(positive),
                          -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.item_knn.knnQuery(embedding)
        return [(self.index_to_item_id[idx], dt) for idx, dt in zip(neighbors,dist)]

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
