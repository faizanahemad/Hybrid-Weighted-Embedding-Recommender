from .recommendation_base import RecommendationBase, Feature, FeatureSet
from typing import List, Dict, Tuple, Sequence, Type, Set
from sklearn.decomposition import PCA
from scipy.special import comb
from pandas import DataFrame
from .content_embedders import ContentEmbedderBase
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
    def __init__(self, embedding_mapper: dict[str, Type[ContentEmbedderBase]],
                 n_dims=32, auto_encoder_layers=2, max_auto_encoder_iter=20):
        super().__init__()
        self.n_dims = n_dims
        self.auto_encoder_layers = auto_encoder_layers
        self.max_auto_encoder_iter = max_auto_encoder_iter
        self.embedding_mapper: dict[str, Type[ContentEmbedderBase]] = embedding_mapper

    def add_users(self, users: List[str]):
        super().add_users(users)

    def add_items(self, items: List[str]):
        super().add_items(items)

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            warm_start=True,
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
        item_data: FeatureSet = kwargs["item_data"]
        user_data: FeatureSet = kwargs["user_data"]
        user_item_affinities: Tuple[List[str], List[str], List[float]] = kwargs["user_item_affinities"]
        user_item_dict: Dict[str, Dict[str, float]] = {}

        item_to_index = dict(zip(item_ids, range(len(item_ids))))

        for user, item, affinity in zip(user_item_affinities[0],user_item_affinities[1],user_item_affinities[2]):
            if user not in user_item_dict:
                user_item_dict[user] = {}
            user_item_dict[user][item] = affinity

        item_embeddings = {}
        user_embeddings = {}
        for feature in item_data:
            if feature.feature_type == "id":
                continue
            feature_name = feature.feature_name
            embedding = self.embedding_mapper[feature_name].fit_predict(feature)
            item_embeddings[feature_name] = embedding

        # Make one ndarray
        # Make user_averaged_item_vectors

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
                final_embedding = (embedding + item_em)/2.0
                user_embedding[i] = final_embedding
            processed_features.append(feature_name)

        for feature in item_data:
            if feature.feature_type == "id" or feature.feature_name in processed_features:
                continue
            feature_name = feature.feature_name
            item_embedding = item_embeddings[feature_name]
            user_embedding = np.zeros(shape=(len(user_ids),item_embedding.shape[1]))
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





        # AutoEncoder them so that error is minimised and distance is maintained
        # https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances
        # Distance Preserving vs Non Preserving











        pass

    def default_predictions(self):
        pass

    def find_similar_items(self, item: str, positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def find_similar_users(self, user: str, positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def find_items_for_user(self, user: List[str], positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
