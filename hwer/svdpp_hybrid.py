import time
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from more_itertools import flatten

from .hybrid_recommender import HybridRecommender
from .logging import getLogger
from .utils import get_rng, normalize_affinity_scores_by_user_item_bs, unit_length


class SVDppHybrid(HybridRecommender):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        assert n_content_dims == n_collaborative_dims
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         n_collaborative_dims, fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)

    def __prediction_network_datagen__(self, user_ids: List[str], item_ids: List[str],
                                       user_item_affinities: List[Tuple[str, str, float]],
                                       user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                       user_vectors: np.ndarray, item_vectors: np.ndarray,
                                       user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                       rating_scale: Tuple[float, float],
                                       batch_size: int, padding_length: int,
                                       noise_augmentation: int, use_content: bool):
        rng = get_rng(noise_augmentation)
        ###

        ratings_count_by_user = Counter([user_id_to_index[u] + 1 for u, i, r in user_item_affinities])
        ratings_count_by_item = Counter([item_id_to_index[i] + 1 for u, i, r in user_item_affinities])

        ratings_count_by_user = defaultdict(int, {k: 1 / np.sqrt(v) for k, v in ratings_count_by_user.items()})
        ratings_count_by_item = defaultdict(int, {k: 1 / np.sqrt(v) for k, v in ratings_count_by_item.items()})

        user_item_list = defaultdict(list)
        item_user_list = defaultdict(list)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1,
                                 ratings_count_by_user[user_id_to_index[u] + 1],
                                 ratings_count_by_item[item_id_to_index[i] + 1], r) for u, i, r in user_item_affinities]
        for i, j, nu, ni, r in user_item_affinities:
            user_item_list[i].append(j)
            item_user_list[j].append(i)
        for k, v in user_item_list.items():
            user_item_list[k] = np.array(v)[:padding_length]

        for k, v in item_user_list.items():
            item_user_list[k] = np.array(v)[:padding_length]

        def gen_fn(i, j, nu, ni):
            user = i
            item = j
            items = user_item_list[i]
            items = np.pad(items, (padding_length - len(items), 0), constant_values=(0, 0)).astype(int)

            users = item_user_list[j]
            users = np.pad(users, (padding_length - len(users), 0), constant_values=(0, 0)).astype(int)

            if use_content:
                ucv = user_content_vectors[user]
                uv = user_vectors[user]
                icv = item_content_vectors[item]
                iv = item_vectors[item]
                return user, item, users, items, nu, ni, ucv, uv, icv, iv
            return user, item, users, items, nu, ni

        def generate_training_samples(affinities):
            def generator():
                for i in range(0, len(affinities), batch_size):
                    start = i
                    end = min(i + batch_size, len(affinities))
                    generated = [(gen_fn(u, v, nu, ni), r + rng(1)) for u, v, nu, ni, r in affinities[start:end]]
                    for g in generated:
                        yield g

            return generator
        return generate_training_samples, gen_fn, ratings_count_by_user, ratings_count_by_item, user_item_list, item_user_list

    def __calculate_bias__(self, user_ids: List[str], item_ids: List[str],
                           user_item_affinities: List[Tuple[str, str, float]],
                           rating_scale: Tuple[float, float],):
        mu, user_bias, item_bias, _, _ = normalize_affinity_scores_by_user_item_bs(user_item_affinities, rating_scale)

        user_bias = np.array([user_bias[u] if u in user_bias else 0.0 for u in user_ids])
        item_bias = np.array([item_bias[i] if i in item_bias else 0.0 for i in item_ids])
        user_bias = np.concatenate(([0], user_bias))
        item_bias = np.concatenate(([0], item_bias))
        return mu, user_bias, item_bias

    def __build_dataset__(self, user_ids: List[str], item_ids: List[str],
                          user_item_affinities: List[Tuple[str, str, float]],
                          user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                          user_vectors: np.ndarray, item_vectors: np.ndarray,
                          user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                          rating_scale: Tuple[float, float], hyperparams: Dict):
        import tensorflow as tf
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        noise_augmentation = hyperparams["noise_augmentation"] if "noise_augmentation" in hyperparams else 0
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else True
        ratings = np.array([r for u, i, r in user_item_affinities])
        min_affinity = np.min(ratings)
        max_affinity = np.max(ratings)
        mu, user_bias, item_bias = self.__calculate_bias__(user_ids, item_ids, user_item_affinities, rating_scale)

        self.log.debug("Mu = %.4f, Max User Bias = %.4f, Max Item Bias = %.4f, min-max-affinity = %s",
                       mu, np.abs(np.max(user_bias)),
                       np.abs(np.max(item_bias)), (min_affinity, max_affinity))

        generate_training_samples, gen_fn, ratings_count_by_user, ratings_count_by_item, user_item_list, item_user_list = self.__prediction_network_datagen__(
            user_ids, item_ids,
            user_item_affinities,
            user_content_vectors,
            item_content_vectors,
            user_vectors, item_vectors,
            user_id_to_index,
            item_id_to_index,
            rating_scale, batch_size, padding_length,
            noise_augmentation, use_content)
        prediction_output_shape = ((), (), padding_length, padding_length, (), ())
        prediction_output_types = (tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64)
        if use_content:
            prediction_output_shape = prediction_output_shape + (self.n_content_dims, self.n_collaborative_dims, self.n_content_dims, self.n_collaborative_dims)
            prediction_output_types = prediction_output_types + (tf.float64, tf.float64, tf.float64, tf.float64)
        output_shapes = (prediction_output_shape, ())
        output_types = (prediction_output_types, tf.float64)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1,
                                 ratings_count_by_user[user_id_to_index[u] + 1],
                                 ratings_count_by_item[item_id_to_index[i] + 1], r) for u, i, r in user_item_affinities]

        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.shuffle(batch_size*4).batch(batch_size).prefetch(2)
        return mu, user_bias, item_bias, train, \
               ratings_count_by_user, ratings_count_by_item, \
               min_affinity, max_affinity, user_item_list, item_user_list, \
               gen_fn, prediction_output_shape, prediction_output_types

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        from tensorflow import keras
        import tensorflow as tf
        import tensorflow.keras.backend as K
        from .tf_utils import LRSchedule, resnet_layer_with_content, root_mean_squared_error, mean_absolute_error
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        n_collaborative_dims = user_vectors.shape[1]
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 128
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        use_resnet = hyperparams["use_resnet"] if "use_resnet" in hyperparams else False

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        user_content_vectors = np.concatenate((np.zeros((1,user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1,item_content_vectors.shape[1])), item_content_vectors))
        user_vectors = np.concatenate((np.zeros((1,user_vectors.shape[1])), user_vectors))
        item_vectors = np.concatenate((np.zeros((1,item_vectors.shape[1])), item_vectors))

        mu, user_bias, item_bias, train, \
        ratings_count_by_user, ratings_count_by_item, \
        min_affinity, \
        max_affinity, user_item_list, item_user_list, \
        gen_fn, prediction_output_shape, prediction_output_types = self.__build_dataset__(user_ids, item_ids,
                                                                              user_item_affinities,
                                                                              user_content_vectors,
                                                                              item_content_vectors,
                                                                              user_vectors, item_vectors,
                                                                              user_id_to_index,
                                                                              item_id_to_index,
                                                                              rating_scale, hyperparams)
        assert np.sum(np.isnan(user_bias)) == 0
        assert np.sum(np.isnan(item_bias)) == 0
        assert np.sum(np.isnan(user_content_vectors)) == 0
        assert np.sum(np.isnan(item_content_vectors)) == 0
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        input_user = keras.Input(shape=(1,), dtype=tf.int64)
        input_item = keras.Input(shape=(1,), dtype=tf.int64)
        input_items = keras.Input(shape=(padding_length,), dtype=tf.int64)
        input_users = keras.Input(shape=(padding_length,), dtype=tf.int64)
        input_nu = keras.Input(shape=(1,))
        input_ni = keras.Input(shape=(1,))

        inputs = [input_user, input_item, input_users, input_items, input_nu, input_ni]
        if use_content:
            input_ucv = keras.Input(shape=(self.n_content_dims,))
            input_uv = keras.Input(shape=(self.n_collaborative_dims,))
            input_icv = keras.Input(shape=(self.n_content_dims,))
            input_iv = keras.Input(shape=(self.n_collaborative_dims,))
            inputs.extend([input_ucv, input_uv, input_icv, input_iv])

        embeddings_initializer = tf.keras.initializers.Constant(user_bias)
        user_bias = keras.layers.Embedding(len(user_ids) + 1, 1, input_length=1, embeddings_initializer=embeddings_initializer)(input_user)

        item_initializer = tf.keras.initializers.Constant(item_bias)
        item_bias = keras.layers.Embedding(len(item_ids) + 1, 1, input_length=1, embeddings_initializer=item_initializer)(input_item)
        user_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_bias)
        item_bias = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_bias)
        user_bias = tf.keras.layers.Flatten()(user_bias)
        item_bias = tf.keras.layers.Flatten()(item_bias)

        def main_network():
            iu = K.concatenate([input_user, input_users])
            user_vecs = keras.layers.Embedding(len(user_ids) + 1, n_collaborative_dims,
                                               input_length=padding_length+1, mask_zero=True)(iu)
            user_vecs = keras.layers.ActivityRegularization(l2=bias_regularizer)(user_vecs)
            user_vec = tf.keras.layers.Lambda(lambda x: x[:, 0])(user_vecs)
            user_vecs = tf.keras.layers.Lambda(lambda x: x[:, 1:])(user_vecs)
            user_vecs = tf.keras.layers.GlobalAveragePooling1D()(user_vecs)
            user_vecs = user_vecs * input_ni

            ii = K.concatenate([input_item, input_items])
            item_initializer = tf.keras.initializers.Constant(item_vectors)
            item_vecs = keras.layers.Embedding(len(item_ids) + 1, n_collaborative_dims,
                                               input_length=padding_length+1, mask_zero=True,
                                               embeddings_initializer=item_initializer)(ii)
            item_vecs = keras.layers.ActivityRegularization(l2=bias_regularizer)(item_vecs)
            item_vec = tf.keras.layers.Lambda(lambda x: x[:, 0])(item_vecs)
            item_vecs = tf.keras.layers.Lambda(lambda x: x[:, 1:])(item_vecs)

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

            if use_content:
                user_item_content_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([input_ucv, input_icv])
                user_item_collab_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([input_uv, input_iv])
                vectors = [input_ucv, input_uv, input_icv, input_iv]
                meta_data = [implicit_term, user_item_vec_dot, item_items_vec_dot, user_user_vec_dot,
                            input_ni, input_nu, user_item_content_similarity, user_item_collab_similarity,
                            user_bias, item_bias]
                vectors = K.concatenate(vectors)
                meta_data = K.concatenate(meta_data)
                meta_data = keras.layers.Dense(network_width, activation="linear", kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(meta_data)
                meta_data = tf.keras.activations.relu(meta_data, alpha=0.1)
                vectors = keras.layers.Dense(4 * self.n_collaborative_dims + 4 * self.n_content_dims, activation="linear", kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(vectors)
                vectors = tf.keras.activations.relu(vectors, alpha=0.1)
                dense_rep = K.concatenate([vectors, meta_data])
                for i in range(network_depth):
                    dense_rep = tf.keras.layers.Dropout(dropout)(dense_rep)
                    if use_resnet:
                        dense_rep = resnet_layer_with_content(network_width, network_width, dropout, kernel_l2)(
                            dense_rep)
                    else:
                        dense_rep = keras.layers.Dense(network_width, activation="linear",
                                                       kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(dense_rep)
                        dense_rep = tf.keras.activations.relu(dense_rep, alpha=0.1)
                    dense_rep = tf.keras.layers.BatchNormalization()(dense_rep)
                rating = keras.layers.Dense(1, activation="linear",
                                            kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(dense_rep)
                rating = tf.keras.activations.relu(rating, alpha=0.1)
                implicit_term = implicit_term + rating

            return implicit_term

        rating = mu + user_bias + item_bias + main_network()

        self.log.debug("Rating Shape = %s", rating.shape)

        model = keras.Model(inputs=inputs, outputs=[rating])

        lr = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=[root_mean_squared_error], metrics=[root_mean_squared_error, mean_absolute_error])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        prediction_artifacts = {"model": model, "user_item_list": user_item_list,
                                "item_user_list": item_user_list,
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size, "gen_fn": gen_fn,
                                "user_content_vectors": user_content_vectors, "item_content_vectors": item_content_vectors,
                                "user_vectors": user_vectors, "item_vectors": item_vectors,
                                "prediction_output_shape": prediction_output_shape,
                                "prediction_output_types": prediction_output_types}
        self.log.info("Built Prediction Network, model params = %s", model.count_params())
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        import tensorflow as tf
        start = time.time()
        model = self.prediction_artifacts["model"]
        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]
        batch_size = self.prediction_artifacts["batch_size"]
        user_item_list = self.prediction_artifacts["user_item_list"]
        item_user_list = self.prediction_artifacts["item_user_list"]
        padding_length = self.prediction_artifacts["padding_length"]
        user_content_vectors = self.prediction_artifacts["user_content_vectors"]
        user_vectors = self.prediction_artifacts["user_vectors"]
        item_content_vectors = self.prediction_artifacts["item_content_vectors"]
        item_vectors = self.prediction_artifacts["item_vectors"]
        gen_fn = self.prediction_artifacts["gen_fn"]
        prediction_output_shape = self.prediction_artifacts["prediction_output_shape"]
        prediction_output_types = self.prediction_artifacts["prediction_output_types"]
        batch_size = max(1024, batch_size)

        def generate_prediction_samples(affinities):
            def generator():
                for i in range(0, len(affinities), batch_size):
                    start = i
                    end = min(i + batch_size, len(affinities))
                    generated = [gen_fn(u, v, nu, ni) for u, v, nu, ni in affinities[start:end]]
                    for g in generated:
                        yield g
            return generator

        if self.fast_inference:
            return self.fast_predict(user_item_pairs)

        if self.super_fast_inference:
            return self.super_fast_predict(user_item_pairs)

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0,
                ratings_count_by_user[self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0],
                ratings_count_by_item[self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0]) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0
        predict = tf.data.Dataset.from_generator(generate_prediction_samples(uip),
                                                 output_types=prediction_output_types, output_shapes=prediction_output_shape, )
        predict = predict.batch(batch_size).prefetch(1)
        predictions = np.array(list(flatten([model.predict(x).reshape((-1)) for x in predict])))
        predictions[np.isnan(predictions)] = [self.mu + self.bu[u] + self.bi[i] for u, i in np.array(user_item_pairs)[np.isnan(predictions)]]
        assert len(predictions) == len(user_item_pairs)
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions

    def prepare_for_knn(self, alpha, content_data_used,
                        user_content_vectors, item_content_vectors,
                        user_vectors, item_vectors):
        assert user_vectors.shape == user_content_vectors.shape
        assert item_vectors.shape == item_content_vectors.shape
        if content_data_used:
            user_content_vectors, item_content_vectors = user_content_vectors * alpha, item_content_vectors * alpha
            user_vectors, item_vectors = user_vectors * (1 - alpha), item_vectors * (1 - alpha)
            user_vectors = user_content_vectors + user_vectors
            item_vectors = item_content_vectors + item_vectors
            self.n_output_dims = user_vectors.shape[1]
            assert user_vectors.shape[1] == item_vectors.shape[1] == self.n_output_dims
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)
        return user_vectors,  item_vectors
