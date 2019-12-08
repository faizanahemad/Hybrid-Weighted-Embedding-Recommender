import time
from collections import Counter
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from bidict import bidict
from more_itertools import flatten
from sklearn.model_selection import StratifiedKFold
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from tensorflow import keras

from .hybrid_recommender import HybridRecommender
from .logging import getLogger
from .recommendation_base import EntityType
from .utils import normalize_affinity_scores_by_user_item, RatingPredRegularization, get_rng, \
    LRSchedule, resnet_layer_with_content, ScaledGlorotNormal, root_mean_squared_error, mean_absolute_error, \
    normalize_affinity_scores_by_user_item_bs


class SVDppDNN(HybridRecommender):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims)
        self.log = getLogger(type(self).__name__)

    def __build_dataset__(self, user_ids: List[str], item_ids: List[str],
                          user_item_affinities: List[Tuple[str, str, float]],
                          user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                          user_vectors: np.ndarray, item_vectors: np.ndarray,
                          user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                          rating_scale: Tuple[float, float], hyperparams: Dict):
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        use_svd = hyperparams["use_svd"] if "use_svd" in hyperparams else False
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        user_content_vectors_mean = np.mean(user_content_vectors)
        item_content_vectors_mean = np.mean(item_content_vectors)
        user_vectors_mean = np.mean(user_vectors)
        item_vectors_mean = np.mean(item_vectors)
        self.log.debug("For rng regularization, user_content_vectors_mean = %s,  item_content_vectors_mean = %s, user_vectors_mean = %s, item_vectors_mean = %s",
                       user_content_vectors_mean, item_content_vectors_mean, user_vectors_mean, item_vectors_mean)

        ###
        ratings = np.array([r for u, i, r in user_item_affinities])
        min_affinity = np.min(ratings)
        max_affinity = np.max(ratings)
        mu, user_bias, item_bias, _, _ = normalize_affinity_scores_by_user_item_bs(user_item_affinities, rating_scale)

        def inverse_fn(user_item_predictions):
            rscaled = np.array([r for u, i, r in user_item_predictions])
            return rscaled

        user_bias = np.array([user_bias[u] if u in user_bias else 0.0 for u in user_ids])
        item_bias = np.array([item_bias[i] if i in item_bias else 0.0 for i in item_ids])
        self.log.debug("Mu = %.4f, Max User Bias = %.4f, Max Item Bias = %.4f, use_svd = %s, min-max-affinity = %s",
                       mu, np.abs(np.max(user_bias)),
                       np.abs(np.max(item_bias)), use_svd, (min_affinity, max_affinity))

        ratings_count_by_user = Counter([u for u, i, r in user_item_affinities])
        ratings_count_by_item = Counter([i for u, i, r in user_item_affinities])

        user_item_list = defaultdict(list)
        item_user_list = defaultdict(list)
        for i, j, r in user_item_affinities:
            user_item_list[i].append(item_id_to_index[j])
            item_user_list[j].append(user_id_to_index[i])

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            def generator():
                for i, j, r in affinities:
                    user = user_id_to_index[i]
                    item = item_id_to_index[j]
                    items = np.array(user_item_list[i])
                    items = items[:padding_length]
                    items = items + 1
                    items = np.pad(items, (padding_length - len(items), 0), constant_values=(0, 0))

                    users = np.array(item_user_list[j])
                    users = users[:padding_length]
                    users = users + 1
                    users = np.pad(users, (padding_length - len(users), 0), constant_values=(0, 0))

                    nu = 1 / np.sqrt(ratings_count_by_user[i])
                    ni = 1 / np.sqrt(ratings_count_by_item[j])
                    yield (user, item, users, items, nu, ni), r

            return generator

        output_shapes = (
            ((), (), padding_length, padding_length, (), ()),
            ())
        output_types = (
            (tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64,),
            tf.float64)

        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size*10).batch(batch_size).prefetch(32)
        return mu, user_bias, item_bias, inverse_fn, train, \
               ratings_count_by_user, ratings_count_by_item, \
               min_affinity, max_affinity, user_item_list, item_user_list

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100

        n_collaborative_dims = user_vectors.shape[1]

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]

        mu, user_bias, item_bias, inverse_fn, train, \
        ratings_count_by_user, ratings_count_by_item, \
        min_affinity, \
        max_affinity, user_item_list, item_user_list = self.__build_dataset__(user_ids, item_ids,
                                                                              user_item_affinities,
                                                                              user_content_vectors,
                                                                              item_content_vectors,
                                                                              user_vectors, item_vectors,
                                                                              user_id_to_index,
                                                                              item_id_to_index,
                                                                              rating_scale, hyperparams)
        input_user = keras.Input(shape=(1,))
        input_item = keras.Input(shape=(1,))
        input_items = keras.Input(shape=(padding_length,))
        input_users = keras.Input(shape=(padding_length,))
        input_nu = keras.Input(shape=(1,))
        input_ni = keras.Input(shape=(1,))

        inputs = [input_user, input_item, input_users, input_items, input_nu, input_ni]

        embeddings_initializer = tf.keras.initializers.Constant(user_bias)
        user_bias = keras.layers.Embedding(len(user_ids), 1, input_length=1, embeddings_initializer=embeddings_initializer)(input_user)

        item_initializer = tf.keras.initializers.Constant(item_bias)
        item_bias = keras.layers.Embedding(len(item_ids), 1, input_length=1, embeddings_initializer=item_initializer)(input_item)
        user_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_bias)
        item_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_bias)
        user_bias = tf.keras.layers.Flatten()(user_bias)
        item_bias = tf.keras.layers.Flatten()(item_bias)

        def main_network():
            embeddings_initializer = tf.keras.initializers.Constant(user_vectors)
            user_vec = keras.layers.Embedding(len(user_ids), n_collaborative_dims, input_length=1)(input_user)

            item_initializer = tf.keras.initializers.Constant(item_vectors)
            item_vec = keras.layers.Embedding(len(item_ids), n_collaborative_dims, input_length=1,
                                              embeddings_initializer=item_initializer)(input_item)

            user_initializer = tf.keras.initializers.Constant(
                np.concatenate((np.array([[0.0] * n_collaborative_dims]), user_vectors), axis=0))
            user_vecs = keras.layers.Embedding(len(user_ids) + 1, n_collaborative_dims,
                                               input_length=padding_length, mask_zero=True)(input_users)
            user_vecs = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_vecs)
            user_vecs = tf.keras.layers.GlobalAveragePooling1D()(user_vecs)
            user_vecs = user_vecs * input_ni

            item_initializer = tf.keras.initializers.Constant(
                np.concatenate((np.array([[0.0] * n_collaborative_dims]), item_vectors), axis=0))
            item_vecs = keras.layers.Embedding(len(item_ids) + 1, n_collaborative_dims,
                                               input_length=padding_length, mask_zero=True,
                                               embeddings_initializer=item_initializer)(input_items)
            item_vecs = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_vecs)
            item_vecs = tf.keras.layers.GlobalAveragePooling1D()(item_vecs)
            item_vecs = item_vecs * input_nu

            user_vec = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_vec)
            item_vec = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_vec)
            user_vec = tf.keras.layers.Flatten()(user_vec)
            item_vec = tf.keras.layers.Flatten()(item_vec)
            user_item_vec_dot = tf.keras.layers.Dot(axes=1, normalize=False)([user_vec, item_vec])
            item_items_vec_dot = tf.keras.layers.Dot(axes=1, normalize=False)([item_vec, item_vecs])
            user_user_vec_dot = tf.keras.layers.Dot(axes=1, normalize=False)([user_vec, user_vecs])
            implicit_term = user_item_vec_dot + item_items_vec_dot + user_user_vec_dot
            return implicit_term

        rating = mu + user_bias + item_bias + main_network()

        self.log.debug("Rating Shape = %s", rating.shape)

        model = keras.Model(inputs=inputs, outputs=[rating])

        sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=[root_mean_squared_error], metrics=[root_mean_squared_error, mean_absolute_error])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        prediction_artifacts = {"model": model, "inverse_fn": inverse_fn, "user_item_list": user_item_list,
                                "item_user_list": item_user_list,
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size,}
        self.log.info("Built Prediction Network, model params = %s", model.count_params())
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        start = time.time()
        model = self.prediction_artifacts["model"]
        inverse_fn = self.prediction_artifacts["inverse_fn"]
        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]
        batch_size = self.prediction_artifacts["batch_size"]
        user_item_list = self.prediction_artifacts["user_item_list"]
        item_user_list = self.prediction_artifacts["item_user_list"]
        padding_length = self.prediction_artifacts["padding_length"]

        def generate_prediction_samples(affinities: List[Tuple[str, str]],
                                        user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                        ratings_count_by_user: Counter, ratings_count_by_item: Counter):
            def generator():
                for i, j in affinities:
                    user_idx = user_id_to_index[i]
                    item_idx = item_id_to_index[j]
                    items = np.array(user_item_list[i])
                    items = items[:padding_length]
                    items = items + 1
                    items = np.pad(items, (padding_length - len(items), 0), constant_values=(0, 0))

                    users = np.array(item_user_list[j])
                    users = users[:padding_length]
                    users = users + 1
                    users = np.pad(users, (padding_length - len(users), 0), constant_values=(0, 0))



                    nu = 1 / np.sqrt(ratings_count_by_user[i])
                    ni = 1 / np.sqrt(ratings_count_by_item[j])
                    yield user_idx, item_idx, users, items, nu, ni

            return generator

        output_shapes = (
            (), (), padding_length, padding_length, (), ())
        output_types = (tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64)
        predict = tf.data.Dataset.from_generator(generate_prediction_samples(user_item_pairs,
                                                                             self.user_id_to_index, self.item_id_to_index,
                                                                             ratings_count_by_user, ratings_count_by_item),
                                                 output_types=output_types, output_shapes=output_shapes, )
        predict = predict.batch(batch_size).prefetch(16)
        predictions = np.array(list(flatten([model.predict(x).reshape((-1)) for x in predict])))
        self.log.debug("Predictions shape = %s", predictions.shape)
        assert len(predictions) == len(user_item_pairs)
        users, items = zip(*user_item_pairs)
        invert_start = time.time()
        predictions = inverse_fn([(u, i, r) for u, i, r in zip(users, items, predictions)])
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        self.log.info("Finished Predicting for n_samples = %s, time taken = %.1f, Invert time = %.1f",
                      len(user_item_pairs),
                      time.time() - start, time.time() - invert_start)
        return predictions
