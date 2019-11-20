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
    normalize_affinity_scores_by_user, normalize_affinity_scores_by_user_item
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, biased: bool = True):
        super().__init__(knn_params=knn_params, rating_scale=rating_scale,
                         n_output_dims=n_content_dims + n_collaborative_dims, biased=biased, bias_type=EntityType.USER)
        self.cb = ContentRecommendation(embedding_mapper, knn_params, rating_scale,
                                        n_content_dims, biased, bias_type=EntityType.USER)
        self.n_content_dims = n_content_dims
        self.n_collaborative_dims = n_collaborative_dims
        self.content_data_used = None
        self.prediction_artifacts = None

    def __entity_entity_affinities_trainer__(self,
                                             entity_ids: List[str],
                                             entity_entity_affinities: List[Tuple[str, str, float]],
                                             entity_id_to_index: Dict[str, int],
                                             vectors: np.ndarray,
                                             n_output_dims: int,
                                             hyperparams: Dict) -> np.ndarray:
        train_affinities, validation_affinities = train_test_split(entity_entity_affinities, test_size=0.5)
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

        train = tf.data.Dataset.from_generator(generate_training_samples(train_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        validation = tf.data.Dataset.from_generator(generate_training_samples(validation_affinities),
                                                    output_types=output_types,
                                                    output_shapes=output_shapes, )
        train = train.shuffle(batch_size).batch(batch_size)
        validation = validation.shuffle(batch_size).batch(batch_size)

        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))

        def build_base_network(embedding_size, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(entity_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            # embeddings_constraint=FixedNorm()
            # embeddings_constraint=tf.keras.constraints.unit_norm(axis=2)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            item = tf.keras.layers.GaussianNoise(0.001 * avg_value)(item)

            for i in range(network_depth + 1):
                if i < network_depth:
                    dense = keras.layers.Dense(embedding_size * network_width, activation="relu",
                                               kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                               activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                    item = dense(item)
                    item = tf.keras.layers.BatchNormalization()(item)
                    item = tf.keras.layers.Dropout(0.05)(item)
                else:
                    dense = keras.layers.Dense(embedding_size, activation="linear", use_bias=False,
                                               kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                               activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                    item = dense(item)
            item = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(item)
            item = K.l2_normalize(item, axis=-1)
            base_network = keras.Model(inputs=i1, outputs=item)
            return base_network

        bn = build_base_network(n_output_dims, vectors)

        item_1 = bn(input_1)
        item_2 = bn(input_2)

        pred = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_2])
        #     pred = K.sum(item_1*item_2, keepdims=True, axis=-1)

        #     pred = pred/2 + 0.5
        #     pred = K.clip(pred, -1, 1)
        pred = K.tanh(pred)
        model = keras.Model(inputs=[input_1, input_2],
                            outputs=[pred])
        #     encoder = tf.keras.Model(input_1, item_1)
        encoder = bn

        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=8, verbose=0, )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)

        return encoder.predict(
            tf.data.Dataset.from_tensor_slices([entity_id_to_index[i] for i in entity_ids]).batch(batch_size))

    def __build_collaborative_embeddings__(self, user_item_affinities: List[Tuple[str, str, float]],
                                           item_item_affinities: List[Tuple[str, str, bool]],
                                           user_user_affinities: List[Tuple[str, str, bool]],
                                           user_ids: List[str], item_ids: List[str],
                                           user_vectors: np.ndarray, item_vectors: np.ndarray,
                                           hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:

        if len(item_item_affinities) > 0:
            item_item_params = {} if "item_item_params" not in hyperparams else hyperparams["item_item_params"]

            item_vectors = self.__entity_entity_affinities_trainer__(entity_ids=item_ids,
                                                                     entity_entity_affinities=item_item_affinities,
                                                                     entity_id_to_index=self.item_id_to_index,
                                                                     vectors=item_vectors,
                                                                     n_output_dims=self.n_content_dims,
                                                                     hyperparams=item_item_params)

        if len(user_user_affinities) > 0:
            user_user_params = {} if "user_user_params" not in hyperparams else hyperparams["user_user_params"]
            user_vectors = self.__entity_entity_affinities_trainer__(entity_ids=user_ids,
                                                                     entity_entity_affinities=user_user_affinities,
                                                                     entity_id_to_index=self.user_id_to_index,
                                                                     vectors=user_vectors,
                                                                     n_output_dims=self.n_content_dims,
                                                                     hyperparams=user_user_params)

        if len(user_item_affinities) > 0:
            user_item_params = {} if "user_item_params" not in hyperparams else hyperparams["user_item_params"]
            user_vectors, item_vectors = self.__user_item_affinities_trainer__(user_ids, item_ids, user_item_affinities,
                                                                               user_vectors, item_vectors,
                                                                               self.user_id_to_index,
                                                                               self.item_id_to_index,
                                                                               self.n_collaborative_dims,
                                                                               user_item_params)
        return user_vectors, item_vectors

    def __user_item_affinities_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_output_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
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

        # max_affinity = np.max(np.abs([r for u, i, r in user_item_affinities]))
        max_affinity = np.max([r for u, i, r in user_item_affinities])
        min_affinity = np.min([r for u, i, r in user_item_affinities])
        user_item_affinities = [(u, i, (2 * 0.9*(r - min_affinity) / (max_affinity - min_affinity)) - 0.9) for u, i, r in
                                user_item_affinities]

        n_input_dims = user_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # user_item_affinities = [(u, i, r / max_affinity) for u, i, r in user_item_affinities]
        train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.5)
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
        train = tf.data.Dataset.from_generator(generate_training_samples(train_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        validation = tf.data.Dataset.from_generator(generate_training_samples(validation_affinities),
                                                    output_types=output_types,
                                                    output_shapes=output_shapes, )

        train = train.shuffle(batch_size).batch(batch_size)
        validation = validation.shuffle(batch_size).batch(batch_size)

        def build_base_network(embedding_size, n_output_dims, vectors):
            avg_value = np.mean(vectors)
            i1 = keras.Input(shape=(1,))

            embeddings_initializer = tf.keras.initializers.Constant(vectors)
            embeddings = keras.layers.Embedding(len(user_ids) + len(item_ids), embedding_size, input_length=1,
                                                embeddings_initializer=embeddings_initializer)
            item = embeddings(i1)
            item = tf.keras.layers.Flatten()(item)
            item = tf.keras.layers.GaussianNoise(0.001 * avg_value)(item)
            embedding_size = max(embedding_size, n_output_dims)
            for i in range(network_depth + 1):
                if i < network_depth:
                    dense = keras.layers.Dense(embedding_size * network_width, activation="relu",
                                               kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                               activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                    item = dense(item)
                    item = tf.keras.layers.BatchNormalization()(item)
                    item = tf.keras.layers.Dropout(0.05)(item)
                else:
                    dense = keras.layers.Dense(n_output_dims, activation="linear", use_bias=False,
                                               kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                               activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                    item = dense(item)
            item = K.l2_normalize(item, axis=-1)
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
        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=4, verbose=0, )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)

        user_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([user_id_to_index[i] for i in user_ids]).batch(batch_size))
        item_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([total_users + item_id_to_index[i] for i in item_ids]).batch(batch_size))
        return user_vectors, item_vectors

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):

        # max_affinity = np.max([r for u, i, r in user_item_affinities])
        # min_affinity = np.min([r for u, i, r in user_item_affinities])
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

        max_affinity = rating_scale[1]
        min_affinity = rating_scale[0]
        n_content_dims = user_content_vectors.shape[1]
        n_collaborative_dims = user_vectors.shape[1]

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        user_item_affinities = [(u, i, (2 * (r - min_affinity) / (max_affinity - min_affinity)) - 1) for u, i, r in
                                user_item_affinities]
        inverse_fn = np.vectorize(lambda r: ((r + 1) / 2) * (max_affinity - min_affinity) + min_affinity)
        mean, bu, bi, _, _ = normalize_affinity_scores_by_user_item(user_item_affinities)
        # print(bu, "\n", bi, "\n", np.max(list(bu.values())), np.max(list(bi.values())))
        # print(mean, np.mean([r for u, i, r in user_item_affinities]), "\n",
        #       max_affinity, np.max([r for u, i, r in user_item_affinities]), "\n",
        #       min_affinity, np.min([r for u, i, r in user_item_affinities]))

        bu = np.array([bu[u] if u in bu else np.random.rand()*0.01 for u in user_ids])
        bi = np.array([bi[i] if i in bi else np.random.rand()*0.01 for i in item_ids])

        ratings_count_by_user = Counter([u for u, i, r in user_item_affinities])
        ratings_count_by_item = Counter([i for u, i, r in user_item_affinities])

        train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.5)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            def generator():
                for i, j, r in affinities:
                    user = user_id_to_index[i]
                    item = item_id_to_index[j]
                    user_content = user_content_vectors[user]
                    item_content = item_content_vectors[item]
                    user_collab = user_vectors[user]
                    item_collab = item_vectors[item]

                    ratings_by_user = np.log1p((ratings_count_by_user[i] + 10.0)/10.0)
                    ratings_by_item = np.log1p((ratings_count_by_item[j] + 10.0)/10.0)
                    yield (user, item, user_content, item_content, user_collab, item_collab,
                           ratings_by_user, ratings_by_item), r

            return generator

        output_shapes = (
            ((), (), (n_content_dims), (n_content_dims), (n_collaborative_dims), (n_collaborative_dims), (), ()),
            ())
        output_types = ((tf.int64, tf.int64, (tf.float64), (tf.float64), (tf.float64), (tf.float64), tf.float64, tf.float64), tf.float64)

        train = tf.data.Dataset.from_generator(generate_training_samples(train_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        validation = tf.data.Dataset.from_generator(generate_training_samples(validation_affinities),
                                                    output_types=output_types,
                                                    output_shapes=output_shapes, )

        train = train.shuffle(batch_size).batch(batch_size)
        validation = validation.shuffle(batch_size).batch(batch_size)

        input_user = keras.Input(shape=(1,))
        input_item = keras.Input(shape=(1,))

        user = tf.keras.layers.Flatten()(input_user)
        item = tf.keras.layers.Flatten()(input_item)

        embeddings_initializer = tf.keras.initializers.Constant(bu)
        user_bias = keras.layers.Embedding(len(user_ids), 1, input_length=1,
                                           embeddings_initializer=embeddings_initializer)(user)

        item_initializer = tf.keras.initializers.Constant(bi)
        item_bias = keras.layers.Embedding(len(item_ids), 1, input_length=1,
                                           embeddings_initializer=item_initializer)(item)

        input_1 = keras.Input(shape=(n_content_dims,))
        input_2 = keras.Input(shape=(n_content_dims,))
        input_3 = keras.Input(shape=(n_collaborative_dims,))
        input_4 = keras.Input(shape=(n_collaborative_dims,))

        user_content = tf.keras.layers.Flatten()(input_1)
        item_content = tf.keras.layers.Flatten()(input_2)
        user_collab = tf.keras.layers.Flatten()(input_3)
        item_collab = tf.keras.layers.Flatten()(input_4)

        user_item_content_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_content, item_content])
        user_item_collab_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_collab, item_collab])
        user_item_content_similarity = tf.keras.layers.Flatten()(user_item_content_similarity)
        user_item_content_similarity = tf.keras.layers.Flatten()(user_item_content_similarity)
        input_5 = keras.Input(shape=(1,))
        input_6 = keras.Input(shape=(1,))

        ratings_by_user = tf.keras.layers.Flatten()(input_5)
        ratings_by_item = tf.keras.layers.Flatten()(input_6)

        user_content = keras.layers.Dense(n_content_dims * network_width, activation="relu",
                                          kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                          activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(user_content)
        item_content = keras.layers.Dense(n_content_dims * network_width, activation="relu",
                                          kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                          activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(item_content)
        user_collab = keras.layers.Dense(n_collaborative_dims * network_width, activation="relu",
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                         activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(user_collab)
        item_collab = keras.layers.Dense(n_collaborative_dims * network_width, activation="relu",
                                         kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                         activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(item_collab)

        vectors = K.concatenate([user_content, item_content, user_collab, item_collab])

        counts_data = keras.layers.Dense(8, activation="relu")(K.concatenate([ratings_by_user, ratings_by_item]))
        meta_data = K.concatenate([counts_data, user_item_content_similarity, user_item_collab_similarity])
        meta_data = keras.layers.Dense(16, activation="relu", )(meta_data)

        dense_representation = K.concatenate([meta_data, vectors])
        dense_representation = tf.keras.layers.BatchNormalization()(dense_representation)
        n_dims = 2*(n_collaborative_dims*4) + 2*(n_content_dims*4) + 16

        for i in range(network_depth):

            dense_representation = keras.layers.Dense(n_dims * network_width, activation="relu",
                                                      kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                                      activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(dense_representation)
            dense_representation = tf.keras.layers.BatchNormalization()(dense_representation)
            dense_representation = tf.keras.layers.Dropout(0.1)(dense_representation)

        dense_representation = keras.layers.Dense(128, activation="relu",
                                                  kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                                  activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))(
            dense_representation)

        rating = keras.layers.Dense(1, activation="tanh", use_bias=False)(dense_representation)
        rating = tf.keras.backend.constant(mean) + user_bias + item_bias + rating
        rating = K.clip(rating, -1.01, 1.01)
        model = keras.Model(inputs=[input_user, input_item, input_1, input_2, input_3, input_4,
                                    input_5, input_6],
                            outputs=[rating])

        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=4, verbose=0, )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)

        prediction_artifacts = {"model": model, "inverse_fn": inverse_fn,
                                "ratings_count_by_user": ratings_count_by_user,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size}
        return prediction_artifacts

        # add regularization to dense layer weights,  add batchnorm

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_item_affinities: List[Tuple[str, str, float]],
            **kwargs):
        user_normalized_affinities = super().fit(user_ids, item_ids, user_item_affinities, **kwargs)

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
                                                                              user_normalized_affinities, self.n_content_dims)
        else:
            user_vectors, item_vectors = np.random.rand((len(user_ids), self.n_content_dims)), np.random.rand(
                (len(item_ids), self.n_content_dims))

        user_content_vectors, item_content_vectors = user_vectors.copy(), item_vectors.copy()
        assert user_content_vectors.shape[1] == item_content_vectors.shape[1] == self.n_content_dims

        collaborative_params = {} if "collaborative_params" not in hyperparameters else hyperparameters["collaborative_params"]
        user_vectors, item_vectors = self.__build_collaborative_embeddings__(user_normalized_affinities,
                                                                             item_item_affinities,
                                                                             user_user_affinities, user_ids, item_ids,
                                                                             user_vectors, item_vectors,
                                                                             collaborative_params)

        user_content_vectors, item_content_vectors = user_content_vectors * alpha, item_content_vectors * alpha
        user_vectors, item_vectors = user_vectors * (1 - alpha), item_vectors * (1 - alpha)
        assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_collaborative_dims
        prediction_network_params = {} if "prediction_network_params" not in collaborative_params else collaborative_params[
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

        _, _ = self.__build_knn__(user_ids, item_ids, user_vectors, item_vectors)
        self.fit_done = True
        return user_vectors, item_vectors

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        model = self.prediction_artifacts["model"]
        inverse_fn = self.prediction_artifacts["inverse_fn"]
        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]

        def generate_prediction_samples(affinities: List[Tuple[str, str]],
                                        user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                        user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                        user_vectors: np.ndarray, item_vectors: np.ndarray,
                                        ratings_count_by_user: Counter, ratings_count_by_item: Counter):
            def generator():
                for i, j in affinities:
                    user_idx = self.user_id_to_index[i]
                    item_idx = self.item_id_to_index[j]

                    user = user_id_to_index[i]
                    item = item_id_to_index[j]

                    user_content = user_content_vectors[user]
                    item_content = item_content_vectors[item]
                    user_collab = user_vectors[user]
                    item_collab = item_vectors[item]

                    user_content = user_content.reshape((-1,len(user_content)))
                    item_content = user_content.reshape((-1, len(item_content)))
                    user_collab = user_content.reshape((-1, len(user_collab)))
                    item_collab = user_content.reshape((-1, len(item_collab)))

                    ratings_by_user = np.log1p((ratings_count_by_user[i] + 10.0) / 10.0)
                    ratings_by_item = np.log1p((ratings_count_by_item[j] + 10.0) / 10.0)
                    yield [np.array([user_idx], dtype=int), np.array([item_idx], dtype=int),
                           user_content, item_content, user_collab, item_collab,
                           np.array([ratings_by_user], dtype=float), np.array([ratings_by_item], dtype=float)]

            return generator

        user_ids = list(set([u for u, i in user_item_pairs]))
        item_ids = list(set([i for u, i in user_item_pairs]))
        user_vectors = self.get_embeddings([(u, EntityType.USER) for u in user_ids])
        item_vectors = self.get_embeddings([(i, EntityType.ITEM) for i in item_ids])

        user_id_to_index = bidict(zip(user_ids, list(range(len(user_ids)))))
        item_id_to_index = bidict(zip(item_ids, list(range(len(item_ids)))))
        user_vectors = np.array(user_vectors)
        item_vectors = np.array(item_vectors)
        assert user_vectors.shape[0]
        if self.content_data_used:
            user_content_vectors = user_vectors[:, :self.n_content_dims]
            item_content_vectors = item_vectors[:, :self.n_content_dims]
            assert user_content_vectors.shape[1] == item_content_vectors.shape[1] == self.n_content_dims
            user_vectors = user_vectors[:, self.n_content_dims:]
            item_vectors = item_vectors[:, self.n_content_dims:]
        else:
            user_content_vectors = user_vectors
            item_content_vectors = item_vectors

        datagen = generate_prediction_samples(user_item_pairs,
                                              user_id_to_index, item_id_to_index,
                                              user_content_vectors, item_content_vectors,
                                              user_vectors, item_vectors,
                                              ratings_count_by_user, ratings_count_by_item)

        predictions = model.predict_generator(datagen(), steps=len(user_item_pairs)).reshape((-1))

        predictions = inverse_fn(predictions)
        predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions

    def find_items_for_user(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                            negative: List[Tuple[str, EntityType]] = None) -> List[Tuple[str, float]]:
        results = super().find_items_for_user(user, positive, negative)
        res, dist = zip(*results)
        ratings = self.predict([(user, i) for i in res])
        return list(sorted(zip(res, ratings), key=operator.itemgetter(1), reverse=True))