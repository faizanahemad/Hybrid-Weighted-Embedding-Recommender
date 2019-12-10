import operator
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
from .utils import RatingPredRegularization, get_rng, \
    LRSchedule, resnet_layer_with_content, ScaledGlorotNormal, root_mean_squared_error, mean_absolute_error, \
    normalize_affinity_scores_by_user_item_bs


class HybridRecommenderSVDpp(HybridRecommender):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims)
        self.log = getLogger(type(self).__name__)

    def __build_svd_model__(self, user_item_affinities, svdpp, rating_scale, user_ids, item_ids, n_folds=1):
        start = time.time()
        models = []
        svd_uv, svd_iv = None, None
        affinities = []
        reader = Reader(rating_scale=rating_scale)
        rng_state = np.random.get_state()
        random_int = np.random.randint(1e8)
        assert n_folds >= 1

        def train_svd(train_affinities, ):
            svd_train = pd.DataFrame(train_affinities)
            svd_train = Dataset.load_from_df(svd_train, reader).build_full_trainset()
            np.random.set_state(rng_state)
            svd_model = SVDpp(random_state=random_int, **svdpp)
            svd_model.fit(svd_train)

            svd_inner_users = [svd_model.trainset.to_inner_uid(u) if svd_model.trainset.knows_user(u) else ""
                               for
                               u in user_ids]
            svd_known_users = [svd_model.trainset.knows_user(u) for u in svd_inner_users]
            svd_uv_1 = np.vstack([svd_model.pu[u] if k else np.random.rand(svdpp['n_factors']) * 0.001 for u, k in
                                  zip(svd_inner_users, svd_known_users)])

            svd_inner_items = [svd_model.trainset.to_inner_iid(i) if svd_model.trainset.knows_item(i) else ""
                               for
                               i in item_ids]
            svd_known_items = [svd_model.trainset.knows_item(i) for i in svd_inner_items]
            svd_iv_1 = np.vstack([svd_model.qi[i] if k else np.random.rand(svdpp['n_factors']) * 0.001 for i, k in
                                  zip(svd_inner_items, svd_known_items)])
            return svd_model, svd_uv_1, svd_iv_1

        if n_folds == 1:
            svd_model, svd_uv, svd_iv = train_svd(user_item_affinities)
            svd_validation = pd.DataFrame(user_item_affinities)
            svd_validation = Dataset.load_from_df(svd_validation, reader).build_full_trainset().build_testset()
            svd_predictions = svd_model.test(svd_validation)
            validation_affinities = [(p.uid, p.iid, p.r_ui - p.est) for p in svd_predictions]
            affinities = validation_affinities
            models.append(svd_model)

        else:
            user_item_affinities = np.array(user_item_affinities)
            users_for_each_rating = np.array([u for u, i, r in user_item_affinities])
            X, y = user_item_affinities, users_for_each_rating
            skf = StratifiedKFold(n_splits=n_folds)
            for train_index, test_index in skf.split(X, y):
                train_affinities, validation_affinities = X[train_index], X[test_index]
                train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
                svd_model, svd_uv_1, svd_iv_1 = train_svd(train_affinities)
                models.append(svd_model)
                if svd_uv is None:
                    svd_uv = svd_uv_1
                else:
                    svd_uv = np.concatenate((svd_uv, svd_uv_1), axis=1)

                if svd_iv is None:
                    svd_iv = svd_iv_1
                else:
                    svd_iv = np.concatenate((svd_iv, svd_iv_1), axis=1)

                validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
                svd_validation = pd.DataFrame(validation_affinities)
                svd_validation = Dataset.load_from_df(svd_validation, reader).build_full_trainset().build_testset()
                svd_predictions = svd_model.test(svd_validation)
                validation_affinities = [(p.uid, p.iid, p.r_ui - p.est) for p in svd_predictions]
                affinities.extend(validation_affinities)

        #
        assert len(models) == n_folds
        self.log.debug("Training %s SVD Models in time = %.1f", len(models), time.time() - start)
        return models, svd_uv, svd_iv, affinities

    def __build_dataset__(self, user_ids: List[str], item_ids: List[str],
                          user_item_affinities: List[Tuple[str, str, float]],
                          user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                          user_vectors: np.ndarray, item_vectors: np.ndarray,
                          user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                          rating_scale: Tuple[float, float], hyperparams: Dict):
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        use_svd = hyperparams["use_svd"] if "use_svd" in hyperparams else False
        svdpp = hyperparams["svdpp"] if "svdpp" in hyperparams else {"n_factors": 8, "n_epochs": 10}
        n_svd_folds = hyperparams["n_svd_folds"] if "n_svd_folds" in hyperparams else 5
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        n_content_dims = user_content_vectors.shape[1]
        n_collaborative_dims = user_vectors.shape[1]

        noise_augmentation = hyperparams["noise_augmentation"] if "noise_augmentation" in hyperparams else False
        rng = get_rng(noise_augmentation)
        user_content_vectors_mean = np.mean(user_content_vectors)
        item_content_vectors_mean = np.mean(item_content_vectors)
        user_vectors_mean = np.mean(user_vectors)
        item_vectors_mean = np.mean(item_vectors)
        self.log.debug(
            "For rng regularization, user_content_vectors_mean = %s,  item_content_vectors_mean = %s, user_vectors_mean = %s, item_vectors_mean = %s",
            user_content_vectors_mean, item_content_vectors_mean, user_vectors_mean, item_vectors_mean)

        if use_svd:
            models, svd_uv, svd_iv, user_item_affinities = self.__build_svd_model__(
                user_item_affinities, svdpp, rating_scale, user_ids, item_ids, n_svd_folds)
            assert len(models) == n_svd_folds
        else:
            models, svd_uv, svd_iv = [], np.zeros((len(user_ids), 1)), np.zeros((len(item_ids), 1))
        n_svd_dims = svd_uv.shape[1]
        assert svd_iv.shape[1] == svd_uv.shape[1]
        user_svd_mean = np.mean(svd_uv)
        item_svd_mean = np.mean(svd_iv)
        ###
        ratings = np.array([r for u, i, r in user_item_affinities])
        min_affinity = np.min(ratings)
        max_affinity = np.max(ratings)
        affinity_range = max_affinity - min_affinity
        # user_item_affinities = [(u, i, (2 * (r - min_affinity) / (max_affinity - min_affinity)) - 1) for u, i, r in
        #                         user_item_affinities]
        mu, user_bias, item_bias, _, _ = normalize_affinity_scores_by_user_item_bs(user_item_affinities, rating_scale)

        def inverse_fn(user_item_predictions):
            rscaled = np.array([r for u, i, r in user_item_predictions])
            # rscaled = ((rscaled + 1) / 2) * (max_affinity - min_affinity) + min_affinity
            if use_svd:
                svd_predictions = np.array(
                    [[model.predict(u, i).est for model in models] for u, i, r in
                     user_item_predictions])
                svd_predictions = np.array(svd_predictions).mean(axis=1)
                rscaled = rscaled + svd_predictions
            return rscaled

        user_bias = np.array([user_bias[u] if u in user_bias else np.random.rand() * 0.01 for u in user_ids])
        item_bias = np.array([item_bias[i] if i in item_bias else np.random.rand() * 0.01 for i in item_ids])
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

        user_item_affinities = list(sorted(user_item_affinities, key=operator.itemgetter(0)))

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

                    user_content = user_content_vectors[user] + rng(n_content_dims, 0.001 * user_content_vectors_mean)
                    item_content = item_content_vectors[item] + rng(n_content_dims, 0.001 * item_content_vectors_mean)
                    user_collab = user_vectors[user] + rng(n_collaborative_dims, 0.001 * user_vectors_mean)
                    item_collab = item_vectors[item] + rng(n_collaborative_dims, 0.001 * item_vectors_mean)
                    r = r + rng(1, 0.01 * affinity_range)

                    ratings_by_user = np.log1p((ratings_count_by_user[i] + 10.0) / 10.0)
                    ratings_by_item = np.log1p((ratings_count_by_item[j] + 10.0) / 10.0)
                    nu = 1 / np.sqrt(ratings_count_by_user[i])
                    ni = 1 / np.sqrt(ratings_count_by_item[j])
                    if use_svd:
                        user_svd = svd_uv[user] + rng(n_svd_dims, 0.001 * user_svd_mean)
                        item_svd = svd_iv[item] + rng(n_svd_dims, 0.001 * item_svd_mean)
                        yield (user, item, users, items, nu, ni, user_content, item_content, user_collab, item_collab,
                               user_svd, item_svd, ratings_by_user, ratings_by_item), r
                    else:
                        yield (user, item, users, items, nu, ni, user_content, item_content, user_collab, item_collab,
                               ratings_by_user, ratings_by_item), r

            return generator

        if use_svd:
            output_shapes = (
                ((), (), padding_length, padding_length, (), (), n_content_dims, n_content_dims, n_collaborative_dims,
                 n_collaborative_dims,
                 n_svd_dims,
                 n_svd_dims, (), ()),
                ())
            output_types = (
                (tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
                 tf.float64, tf.float64,
                 tf.float64,
                 tf.float64, tf.float64),
                tf.float64)
        else:
            output_shapes = (
                ((), (), padding_length, padding_length, (), (), n_content_dims, n_content_dims, n_collaborative_dims,
                 n_collaborative_dims,
                 (), ()),
                ())
            output_types = (
                (tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
                 tf.float64,
                 tf.float64, tf.float64),
                tf.float64)

        s = time.time()
        _ = [i for i in generate_training_samples(user_item_affinities)()]
        e = time.time()
        self.log.debug("Total time to Run Training Generator 1 EPOCH = %.1f", e - s)
        train = tf.data.Dataset.from_generator(generate_training_samples(user_item_affinities),
                                               output_types=output_types, output_shapes=output_shapes, )

        train = train.batch(batch_size).shuffle(32).prefetch(32)
        return mu, user_bias, item_bias, inverse_fn, train, n_svd_dims, \
               ratings_count_by_user, ratings_count_by_item, svd_uv, svd_iv, \
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
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 2
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        rating_regularizer = hyperparams["rating_regularizer"] if "rating_regularizer" in hyperparams else 0.0
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        use_svd = hyperparams["use_svd"] if "use_svd" in hyperparams else False
        use_implicit = hyperparams["use_implicit"] if "use_implicit" in hyperparams else False
        use_dnn = hyperparams["use_dnn"] if "use_dnn" in hyperparams else False
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100

        use_resnet = hyperparams["use_resnet"] if "use_resnet" in hyperparams else False
        resnet_content_each_layer = hyperparams[
            "resnet_content_each_layer"] if "resnet_content_each_layer" in hyperparams else False

        n_content_dims = user_content_vectors.shape[1]
        n_collaborative_dims = user_vectors.shape[1]

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]

        mu, user_bias, item_bias, inverse_fn, train, \
        n_svd_dims, ratings_count_by_user, ratings_count_by_item, \
        svd_uv, svd_iv, min_affinity, \
        max_affinity, user_item_list, item_user_list = self.__build_dataset__(user_ids, item_ids,
                                                                              user_item_affinities,
                                                                              user_content_vectors,
                                                                              item_content_vectors,
                                                                              user_vectors, item_vectors,
                                                                              user_id_to_index,
                                                                              item_id_to_index,
                                                                              rating_scale, hyperparams)
        assert svd_uv.shape[1] == svd_iv.shape[1] == n_svd_dims
        self.log.debug("DataSet Built with n_svd_dims = %s, use_svd = %s", n_svd_dims, use_svd)
        input_user = keras.Input(shape=(1,))
        input_item = keras.Input(shape=(1,))
        input_users = keras.Input(shape=(padding_length,))
        input_items = keras.Input(shape=(padding_length,))
        input_nu = keras.Input(shape=(1,))
        input_ni = keras.Input(shape=(1,))

        input_1 = keras.Input(shape=(n_content_dims,))
        input_2 = keras.Input(shape=(n_content_dims,))
        input_3 = keras.Input(shape=(n_collaborative_dims,))
        input_4 = keras.Input(shape=(n_collaborative_dims,))
        input_5 = keras.Input(shape=(1,))
        input_6 = keras.Input(shape=(1,))
        inputs = [input_user, input_item, input_users, input_items,
                  input_nu, input_ni, input_1, input_2, input_3, input_4, input_5, input_6]
        if use_svd:
            input_svd_uv = keras.Input(shape=(n_svd_dims,))
            input_svd_iv = keras.Input(shape=(n_svd_dims,))
            inputs = [input_user, input_item, input_users, input_items,
                      input_nu, input_ni, input_1, input_2, input_3, input_4, input_svd_uv,
                      input_svd_iv,
                      input_5, input_6]

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

        def main_network():
            initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1)
            embeddings_initializer = tf.keras.initializers.Constant(user_vectors)
            user_vec = keras.layers.Embedding(len(user_ids), n_collaborative_dims, input_length=1)(input_user)

            item_initializer = tf.keras.initializers.Constant(item_vectors)
            item_vec = keras.layers.Embedding(len(item_ids), n_collaborative_dims, input_length=1,
                                              embeddings_initializer=initializer)(input_item)

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
                                               embeddings_initializer=initializer)(input_items)
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

            user_content = input_1
            item_content = input_2
            user_collab = input_3
            item_collab = input_4

            user_item_content_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_content, item_content])
            user_item_collab_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([user_collab, item_collab])

            ratings_by_user = input_5
            ratings_by_item = input_6

            vectors = [user_content, item_content, user_collab, item_collab, user_vec, item_vec, item_vecs, user_vecs]
            counts_data = keras.layers.Dense(8, activation="tanh", use_bias=False)(
                K.concatenate([ratings_by_user, ratings_by_item, input_nu, input_ni, ]))
            meta_data = [counts_data, user_item_content_similarity, user_item_collab_similarity,
                         item_bias, user_bias, implicit_term, item_items_vec_dot, user_item_vec_dot, user_user_vec_dot]
            if use_svd:
                user_svd = input_svd_uv
                item_svd = input_svd_iv
                user_item_svd_similarity = tf.keras.layers.Dot(axes=1, normalize=False)([user_svd, item_svd])
                vectors.extend([user_svd, item_svd])
                meta_data.append(user_item_svd_similarity)

            vectors = K.concatenate(vectors)
            meta_data = K.concatenate(meta_data)
            meta_data = keras.layers.Dense(64, activation="tanh", )(meta_data)

            dense_representation = K.concatenate([meta_data, vectors])
            # dense_representation = tf.keras.layers.BatchNormalization()(dense_representation)
            initial_dense_representation = dense_representation if resnet_content_each_layer else None
            self.log.info("Start Training: lr = %.5f, use_dnn = %s, use_implicit = %s,use_svd = %s, " +
                          "use_resnet = %s, dense_dims = %s, vector shape = %s, " +
                          "network_depth = %s, network width = %s, dropout = %.2f, ",
                          lr, use_dnn, use_implicit, use_svd, use_resnet,
                          dense_representation.shape, vectors.shape,
                          network_depth, network_width, dropout)

            dims_fn = lambda x: int(
                np.interp(x, [0, network_depth - 1], [dense_representation.shape[1] * 2, network_width]))
            for i in range(0, network_depth):
                dims = dims_fn(i)
                if use_resnet:
                    dense_representation = resnet_layer_with_content(dims, dims, dropout, kernel_l2)(
                        dense_representation, initial_dense_representation)

                else:
                    dense_representation = keras.layers.Dense(dims, activation="tanh", use_bias=False,
                                                              kernel_initializer=ScaledGlorotNormal(),
                                                              kernel_regularizer=keras.regularizers.l1_l2(
                                                                  l2=kernel_l2))(
                        dense_representation)
                    # dense_representation = tf.keras.layers.BatchNormalization()(dense_representation)
                    dense_representation = tf.keras.layers.Dropout(dropout)(dense_representation)

            rating = keras.layers.Dense(1, activation="linear", use_bias=False, kernel_initializer=ScaledGlorotNormal(),
                                        kernel_regularizer=keras.regularizers.l1_l2(l2=kernel_l2))(
                dense_representation)
            result = 0.0
            if use_implicit:
                result += implicit_term
            if use_dnn:
                rating = keras.layers.ActivityRegularization(l2=bias_regularizer)(rating)
                result += rating
            return result

        rating = tf.keras.backend.constant(mu) + user_bias + item_bias + main_network()
        self.log.debug("Before Rating Regularization, min-max affinity for DNN = %s", (min_affinity, max_affinity))
        rating = RatingPredRegularization(l2=rating_regularizer, min_r=min_affinity, max_r=max_affinity)(rating)

        model = keras.Model(inputs=inputs, outputs=[rating])

        # lr = LRSchedule(lr=lr, epochs=epochs, batch_size=batch_size, n_examples=len(user_item_affinities))
        sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=[root_mean_squared_error], metrics=[root_mean_squared_error, mean_absolute_error])

        model.fit(train, epochs=epochs, callbacks=[], verbose=verbose)

        prediction_artifacts = {"model": model, "inverse_fn": inverse_fn, "user_item_list": user_item_list,
                                "item_user_list": item_user_list,
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size, "svd_uv": svd_uv, "svd_iv": svd_iv, "use_svd": use_svd}
        self.log.info("Built Prediction Network, model params = %s", model.count_params())
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        start = time.time()
        model = self.prediction_artifacts["model"]
        inverse_fn = self.prediction_artifacts["inverse_fn"]
        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]
        svd_uv = self.prediction_artifacts["svd_uv"]
        svd_iv = self.prediction_artifacts["svd_iv"]
        batch_size = self.prediction_artifacts["batch_size"]
        use_svd = self.prediction_artifacts["use_svd"]
        user_item_list = self.prediction_artifacts["user_item_list"]
        item_user_list = self.prediction_artifacts["item_user_list"]
        padding_length = self.prediction_artifacts["padding_length"]

        def generate_prediction_samples(affinities: List[Tuple[str, str]],
                                        global_user_id_to_index: Dict[str, int],
                                        global_item_id_to_index: Dict[str, int],
                                        user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                        user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                        user_vectors: np.ndarray, item_vectors: np.ndarray,
                                        svd_uv: np.ndarray, svd_iv: np.ndarray,
                                        ratings_count_by_user: Counter, ratings_count_by_item: Counter):
            def generator():
                for i, j in affinities:
                    user_idx = global_user_id_to_index[i]
                    item_idx = global_item_id_to_index[j]
                    items = np.array(user_item_list[i])
                    items = items[:padding_length]
                    items = items + 1
                    items = np.pad(items, (padding_length - len(items), 0), constant_values=(0, 0))

                    users = np.array(item_user_list[j])
                    users = users[:padding_length]
                    users = users + 1
                    users = np.pad(users, (padding_length - len(users), 0), constant_values=(0, 0))

                    user = user_id_to_index[i]
                    item = item_id_to_index[j]
                    user_content = user_content_vectors[user]
                    item_content = item_content_vectors[item]
                    user_collab = user_vectors[user]
                    item_collab = item_vectors[item]
                    nu = 1 / np.sqrt(ratings_count_by_user[i])
                    ni = 1 / np.sqrt(ratings_count_by_item[j])
                    ratings_by_user = np.log1p((ratings_count_by_user[i] + 10.0) / 10.0)
                    ratings_by_item = np.log1p((ratings_count_by_item[j] + 10.0) / 10.0)
                    if use_svd:
                        user_svd = svd_uv[user_idx]
                        item_svd = svd_iv[item_idx]
                        yield user_idx, item_idx, users, items, nu, ni, user_content, item_content, \
                              user_collab, item_collab, user_svd, item_svd, \
                              ratings_by_user, ratings_by_item
                    else:
                        yield user_idx, item_idx, users, items, nu, ni, user_content, item_content, \
                              user_collab, item_collab, \
                              ratings_by_user, ratings_by_item

            return generator

        user_ids = list(set([u for u, i in user_item_pairs]))
        item_ids = list(set([i for u, i in user_item_pairs]))
        user_vectors = self.get_embeddings([(u, EntityType.USER) for u in user_ids])
        item_vectors = self.get_embeddings([(i, EntityType.ITEM) for i in item_ids])

        user_id_to_index = bidict(zip(user_ids, list(range(len(user_ids)))))
        item_id_to_index = bidict(zip(item_ids, list(range(len(item_ids)))))
        user_vectors = np.array(user_vectors)
        item_vectors = np.array(item_vectors)
        assert user_vectors.shape[0] == len(user_ids)
        assert item_vectors.shape[0] == len(item_ids)
        if self.content_data_used:
            user_content_vectors = user_vectors[:, :self.n_content_dims]
            item_content_vectors = item_vectors[:, :self.n_content_dims]
            assert user_content_vectors.shape[1] == item_content_vectors.shape[1] == self.n_content_dims
            user_vectors = user_vectors[:, self.n_content_dims:]
            item_vectors = item_vectors[:, self.n_content_dims:]
        else:
            user_content_vectors = user_vectors
            item_content_vectors = item_vectors

        n_svd_dims = svd_uv.shape[1]
        if use_svd:
            output_shapes = (
                (), (), padding_length, padding_length, (), (), self.n_content_dims, self.n_content_dims,
                self.n_collaborative_dims,
                self.n_collaborative_dims, n_svd_dims, n_svd_dims, (), ())
            output_types = (
            tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
            tf.float64, tf.float64, tf.float64, tf.float64, tf.float64)
        else:
            output_shapes = (
                (), (), padding_length, padding_length, (), (), self.n_content_dims, self.n_content_dims,
                self.n_collaborative_dims,
                self.n_collaborative_dims, (), ())
            output_types = (
            tf.int64, tf.int64, tf.int64, tf.int64, tf.float64, tf.float64, tf.float64, tf.float64, tf.float64,
            tf.float64, tf.float64, tf.float64)
        predict = tf.data.Dataset.from_generator(generate_prediction_samples(user_item_pairs,
                                                                             self.user_id_to_index,
                                                                             self.item_id_to_index,
                                                                             user_id_to_index, item_id_to_index,
                                                                             user_content_vectors, item_content_vectors,
                                                                             user_vectors, item_vectors, svd_uv, svd_iv,
                                                                             ratings_count_by_user,
                                                                             ratings_count_by_item),
                                                 output_types=output_types, output_shapes=output_shapes, )
        predict = predict.batch(batch_size).prefetch(16)
        model_start_time = time.time()
        predictions = np.array(list(flatten([model.predict(x).reshape((-1)) for x in predict])))
        model_end_time = time.time()
        model_time = model_end_time - model_start_time
        assert len(predictions) == len(user_item_pairs)
        users, items = zip(*user_item_pairs)
        invert_start = time.time()
        predictions = inverse_fn([(u, i, r) for u, i, r in zip(users, items, predictions)])
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        self.log.debug(
            "Finished Predicting for n_samples = %s, time taken = %.2f, Model Time Taken = %.2f, Invert time = %.2f",
            len(user_item_pairs), time.time() - start,
            model_time, time.time() - invert_start)
        return predictions
