from .recommendation_base import RecommendationBase, Feature, FeatureSet
from .logging import getLogger
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
    normalize_affinity_scores_by_user, normalize_affinity_scores_by_user_item, UnitLengthRegularization
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from scipy.stats import describe
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import Reader

from .hybrid_recommender import HybridRecommender
from .hybrid_recommender_svdpp import HybridRecommenderSVDpp


class HybridRecommenderTripletLoss(HybridRecommenderSVDpp):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims)
        self.log = getLogger(type(self).__name__)

    def __entity_entity_affinities_trainer__(self,
                                             entity_ids: List[str],
                                             entity_entity_affinities: List[Tuple[str, str, float]],
                                             entity_id_to_index: Dict[str, int],
                                             vectors: np.ndarray,
                                             n_output_dims: int,
                                             hyperparams: Dict) -> np.ndarray:
        self.log.debug("Start Training Entity Affinities, n_entities = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(entity_ids), len(entity_entity_affinities), vectors.shape, n_output_dims)
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
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.1
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.25
        random_positive_weight = hyperparams["random_positive_weight"] if "random_positive_weight" in hyperparams else 0.1
        random_negative_weight = hyperparams["random_negative_weight"] if "random_negative_weight" in hyperparams else 0.25
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.1

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            item_close_dict = {}
            item_far_dict = {}
            for i, j, r in affinities:
                assert r != 0
                if r > 0:
                    if i in item_close_dict:
                        item_close_dict[i].append((j, r))
                    else:
                        item_close_dict[i] = [(j, r)]

                    if j in item_close_dict:
                        item_close_dict[j].append((i, r))
                    else:
                        item_close_dict[j] = [(i, r)]
                if r < 0:
                    if i in item_far_dict:
                        item_far_dict[i].append((j, r))
                    else:
                        item_far_dict[i] = [(j, r)]

                    if j in item_far_dict:
                        item_far_dict[j].append((i, r))
                    else:
                        item_far_dict[j] = [(i, r)]
            total_items = len(entity_ids)
            def generator():
                for i, j, r in affinities:
                    first_item = entity_id_to_index[i]
                    second_item = entity_id_to_index[j]
                    random_item = entity_id_to_index[entity_ids[np.random.randint(0, total_items)]]
                    choose_random_pair = np.random.rand() < random_pair_proba
                    if r < 0:
                        distant_item = second_item
                        distant_item_weight = r

                        if choose_random_pair or i not in item_close_dict:
                            second_item, close_item_weight = random_item, random_positive_weight

                        else:
                            second_item, close_item_weight = item_close_dict[i][np.random.randint(0, len(item_close_dict[i]))]
                            second_item = entity_id_to_index[second_item]
                    else:
                        close_item_weight = r
                        if choose_random_pair or i not in item_far_dict:
                            distant_item, distant_item_weight = random_item, random_negative_weight

                        else:
                            distant_item, distant_item_weight = item_far_dict[i][np.random.randint(0, len(item_far_dict[i]))]
                            distant_item = entity_id_to_index[distant_item]

                    yield (first_item, second_item, distant_item, close_item_weight, distant_item_weight), r
            return generator

        output_shapes = (((), (), (), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64, tf.float32, tf.float32), tf.float32)

        train = tf.data.Dataset.from_generator(generate_training_samples(train_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )
        validation = tf.data.Dataset.from_generator(generate_training_samples(validation_affinities),
                                                    output_types=output_types,
                                                    output_shapes=output_shapes, )
        train = train.shuffle(batch_size).batch(batch_size)
        validation = validation.shuffle(batch_size).batch(batch_size)

        input_1 = keras.Input(shape=(1,))
        input_2 = keras.Input(shape=(1,))
        input_3 = keras.Input(shape=(1,))

        close_weight = keras.Input(shape=(1,))
        far_weight = keras.Input(shape=(1,))

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

            for i in range(network_depth):

                dense = keras.layers.Dense(embedding_size * network_width, activation="relu",
                                           kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                           activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                item = dense(item)
                item = tf.keras.layers.BatchNormalization()(item)
                item = tf.keras.layers.Dropout(dropout)(item)

            dense = keras.layers.Dense(embedding_size, activation="linear", use_bias=False,
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                       activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
            item = dense(item)
            item = UnitLengthRegularization(l1=0.0001, l2=0.001)(item)
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

        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)
        self.log.debug("End Training Entity Affinities")

        return encoder.predict(
            tf.data.Dataset.from_tensor_slices([entity_id_to_index[i] for i in entity_ids]).batch(batch_size))

    def __user_item_affinities_trainer__(self,
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
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 2
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l1 = hyperparams["kernel_l1"] if "kernel_l1" in hyperparams else 0.001
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.001
        activity_l1 = hyperparams["activity_l1"] if "activity_l1" in hyperparams else 0.0005
        activity_l2 = hyperparams["activity_l2"] if "activity_l2" in hyperparams else 0.0005
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.1
        random_pair_proba = hyperparams["random_pair_proba"] if "random_pair_proba" in hyperparams else 0.25
        random_pair_user_item_proba = hyperparams["random_pair_user_item_proba"] if "random_pair_user_item_proba" in hyperparams else 0.25
        random_positive_weight = hyperparams["random_positive_weight"] if "random_positive_weight" in hyperparams else 0.1
        random_negative_weight = hyperparams["random_negative_weight"] if "random_negative_weight" in hyperparams else 0.2
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.1

        # max_affinity = np.max(np.abs([r for u, i, r in user_item_affinities]))
        max_affinity = np.max([r for u, i, r in user_item_affinities])
        min_affinity = np.min([r for u, i, r in user_item_affinities])
        user_item_affinities = [(u, i, (4 * (r - min_affinity) / (max_affinity - min_affinity)) - 2) for u, i, r in
                                user_item_affinities]

        n_input_dims = user_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # user_item_affinities = [(u, i, r / max_affinity) for u, i, r in user_item_affinities]
        train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.5)
        total_users = len(user_ids)
        total_items = len(item_ids)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            user_close_dict = {}
            user_far_dict = {}
            item_close_dict = {}
            item_far_dict = {}
            for i, j, r in affinities:
                assert r != 0
                if r > 0:
                    if i in user_close_dict:
                        user_close_dict[i].append((j, r))
                    else:
                        user_close_dict[i] = [(j, r)]

                    if j in item_close_dict:
                        item_close_dict[j].append((i, r))
                    else:
                        item_close_dict[j] = [(i, r)]
                if r < 0:
                    if i in user_far_dict:
                        user_far_dict[i].append((j, r))
                    else:
                        user_far_dict[i] = [(j, r)]

                    if j in item_far_dict:
                        item_far_dict[j].append((i, r))
                    else:
                        item_far_dict[j] = [(i, r)]

            def generator():
                for i, j, r in affinities:
                    user = user_id_to_index[i]
                    second_item = total_users + item_id_to_index[j]
                    random_item = total_users + item_id_to_index[item_ids[np.random.randint(0, total_items)]]
                    random_user = user_id_to_index[user_ids[np.random.randint(0, total_users)]]
                    choose_random_pair = np.random.rand() < random_pair_proba
                    choose_user_pair = np.random.rand() < random_pair_user_item_proba
                    if r < 0:
                        distant_item = second_item
                        distant_item_weight = r

                        if choose_random_pair or (i not in user_close_dict and j not in item_close_dict):
                            second_item, close_item_weight = random_user if choose_user_pair else random_item, random_positive_weight
                        else:
                            if (choose_user_pair and j in item_close_dict) or i not in user_close_dict:
                                second_item, close_item_weight = item_close_dict[j][np.random.randint(0, len(item_close_dict[j]))]
                                second_item = total_users + user_id_to_index[second_item]
                            else:
                                second_item, close_item_weight = user_close_dict[i][np.random.randint(0, len(user_close_dict[i]))]
                                second_item = item_id_to_index[second_item]
                    else:
                        close_item_weight = r
                        if choose_random_pair or (i not in user_far_dict and j not in item_far_dict):
                            distant_item, distant_item_weight = random_user if choose_user_pair else random_item, random_negative_weight
                        else:
                            if (choose_user_pair and j in item_far_dict) or i not in user_far_dict:
                                distant_item, distant_item_weight = item_far_dict[j][np.random.randint(0, len(item_far_dict[j]))]
                                distant_item = total_users + user_id_to_index[distant_item]
                            else:
                                distant_item, distant_item_weight = user_far_dict[i][np.random.randint(0, len(user_far_dict[i]))]
                                distant_item = item_id_to_index[distant_item]

                    yield (user, second_item, distant_item, close_item_weight, distant_item_weight), r
            return generator

        output_shapes = (((), (), (), (), ()), ())
        output_types = ((tf.int64, tf.int64, tf.int64, tf.float32, tf.float32), tf.float32)

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
            for i in range(network_depth):

                dense = keras.layers.Dense(embedding_size * network_width, activation="relu",
                                           kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                           activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
                item = dense(item)
                item = tf.keras.layers.BatchNormalization()(item)
                item = tf.keras.layers.Dropout(dropout)(item)

            dense = keras.layers.Dense(n_output_dims, activation="linear", use_bias=False,
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=kernel_l1, l2=kernel_l2),
                                       activity_regularizer=keras.regularizers.l1_l2(l1=activity_l1, l2=activity_l2))
            item = dense(item)
            item = UnitLengthRegularization(l1=0.0001, l2=0.001)(item)
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
        i1_i2_dist = close_weight * i1_i2_dist

        i1_i3_dist = tf.keras.layers.Dot(axes=1, normalize=True)([item_1, item_3])
        i1_i3_dist = 1 - i1_i3_dist
        i1_i3_dist = i1_i3_dist / K.abs(far_weight)

        loss = K.relu(i1_i2_dist - i1_i3_dist + margin)
        model = keras.Model(inputs=[input_1, input_2, input_3, close_weight, far_weight],
                            outputs=[loss])

        encoder = bn
        adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
        model.compile(optimizer=adam,
                      loss=['mean_squared_error'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(train, epochs=epochs,
                  validation_data=validation, callbacks=callbacks, verbose=verbose)

        K.set_value(model.optimizer.lr, lr)

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0,
                                              restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001)
        callbacks = [es, reduce_lr]

        model.fit(validation, epochs=epochs,
                  validation_data=train, callbacks=callbacks, verbose=verbose)

        user_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([user_id_to_index[i] for i in user_ids]).batch(batch_size))
        item_vectors = encoder.predict(
            tf.data.Dataset.from_tensor_slices([total_users + item_id_to_index[i] for i in item_ids]).batch(batch_size))
        self.log.debug("End Training User-Item Affinities")
        return user_vectors, item_vectors
