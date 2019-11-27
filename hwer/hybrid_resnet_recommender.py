from .recommendation_base import RecommendationBase, Feature, FeatureSet
from typing import List, Dict, Tuple, Sequence, Type, Set, Optional
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
from bidict import bidict
from joblib import Parallel, delayed
from collections import defaultdict
import gc
import sys
import os
from more_itertools import flatten
import dill
from collections import Counter
import operator
from tqdm import tqdm_notebook
import fasttext
from .recommendation_base import EntityType
from .content_recommender import ContentRecommendation
from .utils import unit_length, build_user_item_dict, build_item_user_dict, cos_sim, shuffle_copy, \
    normalize_affinity_scores_by_user, normalize_affinity_scores_by_user_item, RatingPredRegularization
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from scipy.stats import describe
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import Reader

from .hybrid_recommender import HybridRecommender
from .hybrid_triplet_loss_recommender import HybridRecommenderTripletLoss
from .hybrid_recommender_svdpp import HybridRecommenderSVDpp
from .hybrid_recommender_svdpp import HybridRecommenderSVDpp


def resnet_layer_with_content(n_dims, n_out_dims, dropout, activity_l2, kernel_l2, depth=2):
    assert n_dims >= n_out_dims

    def layer(x, content=None):
        if content is not None:
            h = K.concatenate([x, content])
        else:
            h = x
        for i in range(1, depth + 1):
            dims = n_dims if i < depth else n_out_dims
            h = keras.layers.Dense(dims, activation="tanh",
                                   kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2),
                                   activity_regularizer=keras.regularizers.l1_l2(l2=activity_l2))(h)
            h = tf.keras.layers.BatchNormalization()(h)
            h = tf.keras.layers.Dropout(dropout)(h)
        if x.shape[1] != n_out_dims:
            x = keras.layers.Dense(n_out_dims, activation="linear",
                                   kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2),
                                   activity_regularizer=keras.regularizers.l1_l2(l2=activity_l2))(x)
        return h + x
    return layer


class HybridRecommenderResnet(HybridRecommenderSVDpp):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims)

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 2
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l1 = hyperparams["kernel_l1"] if "kernel_l1" in hyperparams else 0.001
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001
        activity_l1 = hyperparams["activity_l1"] if "activity_l1" in hyperparams else 0.0005
        activity_l2 = hyperparams["activity_l2"] if "activity_l2" in hyperparams else 0.0005
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.01
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.1
        resnet_layers = hyperparams["residual_layers"] if "residual_layers" in hyperparams else 4
        resnet_width = hyperparams["resnet_width"] if "resnet_width" in hyperparams else 128
        resnet_content_each_layer = hyperparams["resnet_content_each_layer"] if "resnet_content_each_layer" in hyperparams else False

        n_content_dims = user_content_vectors.shape[1]
        n_collaborative_dims = user_vectors.shape[1]

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]

        mu, user_bias, item_bias, inverse_fn, train, validation, \
        n_svd_dims, ratings_count_by_user, ratings_count_by_item, \
        svd_uv, svd_iv = self.__build_dataset__(user_ids, item_ids, user_item_affinities,
                                                user_content_vectors, item_content_vectors,
                                                user_vectors, item_vectors,
                                                user_id_to_index, item_id_to_index,
                                                rating_scale, hyperparams)
        input_user = keras.Input(shape=(1,))
        input_item = keras.Input(shape=(1,))

        embeddings_initializer = tf.keras.initializers.Constant(user_bias)
        user_bias = keras.layers.Embedding(len(user_ids), 1, input_length=1,
                                           embeddings_initializer=embeddings_initializer)(input_user)

        item_initializer = tf.keras.initializers.Constant(item_bias)
        item_bias = keras.layers.Embedding(len(item_ids), 1, input_length=1,
                                           embeddings_initializer=item_initializer)(input_item)
        user_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_bias)
        item_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_bias)
        user_bias = tf.keras.layers.Flatten()(user_bias)
        item_bias = tf.keras.layers.Flatten()(item_bias)

        input_1 = keras.Input(shape=(n_content_dims,))
        input_2 = keras.Input(shape=(n_content_dims,))
        input_3 = keras.Input(shape=(n_collaborative_dims,))
        input_4 = keras.Input(shape=(n_collaborative_dims,))

        input_svd_uv = keras.Input(shape=(n_svd_dims,))
        input_svd_iv = keras.Input(shape=(n_svd_dims,))

        user_content = input_1
        item_content = input_2
        user_collab = input_3
        item_collab = input_4
        user_svd = input_svd_uv
        item_svd = input_svd_iv

        user_item_content_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_content, item_content])
        user_item_collab_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_collab, item_collab])
        user_item_svd_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_svd, item_svd])

        input_5 = keras.Input(shape=(1,))
        input_6 = keras.Input(shape=(1,))

        ratings_by_user = input_5
        ratings_by_item = input_6

        vectors = K.concatenate([user_content, item_content, user_collab, item_collab, user_svd, item_svd])
        meta_data = K.concatenate([ratings_by_item, ratings_by_user,
                                   user_item_content_similarity, user_item_collab_similarity, user_item_svd_similarity,
                                   item_bias, user_bias])
        meta_data = keras.layers.Dense(16 * network_width, activation="tanh", )(meta_data)

        dense_representation = K.concatenate([meta_data, vectors])

        print("RESNET REC: dense shape = ", dense_representation.shape, ", resnet width =", resnet_width,
              ", content based resnet or not = ", resnet_content_each_layer)

        dense_representation = keras.layers.Dense(resnet_width, activation="tanh",
                                                  kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1,
                                                                                              l2=kernel_l2),
                                                  activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1,
                                                                                                l2=activity_l2))(
            dense_representation)
        dense_representation = tf.keras.layers.BatchNormalization()(dense_representation)
        initial_dense_representation = dense_representation if resnet_content_each_layer else None

        for i in range(1, resnet_layers+1):
            dense_representation = resnet_layer_with_content(resnet_width, resnet_width, dropout, activity_l2, kernel_l2)(dense_representation, initial_dense_representation)

        dense_representation = keras.layers.Dense(int(resnet_width / 2), activation="tanh",
                                                  kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1,
                                                                                              l2=kernel_l2),
                                                  activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1,
                                                                                                l2=activity_l2))(
            dense_representation)

        rating = keras.layers.Dense(1, activation="tanh", use_bias=True,
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                    activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(
            dense_representation)
        rating = tf.keras.backend.constant(mu) + user_bias + item_bias + rating
        rating = RatingPredRegularization(l1=0.01, l2=0.001)(rating)
        # rating = K.clip(rating, -1.0, 1.0)
        model = keras.Model(
            inputs=[input_user, input_item, input_1, input_2, input_3, input_4, input_svd_uv, input_svd_iv,
                    input_5, input_6],
            outputs=[rating])

        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)

        full_dataset = validation.unbatch().concatenate(train.unbatch()).shuffle(batch_size).batch(batch_size)
        model.fit(full_dataset, epochs=1, verbose=verbose)
        # print("Train Loss = ", model.evaluate(train), "validation Loss = ", model.evaluate(validation))

        prediction_artifacts = {"model": model, "inverse_fn": inverse_fn,
                                "ratings_count_by_user": ratings_count_by_user,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size, "svd_uv": svd_uv, "svd_iv": svd_iv}
        return prediction_artifacts
