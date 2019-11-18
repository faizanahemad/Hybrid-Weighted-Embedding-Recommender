from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import nmslib
import time
from tqdm import tqdm_notebook
from typing import List, Dict, Tuple, Sequence, Type, Set
from fasttext import FastText
from scipy.special import comb
from sklearn.utils import shuffle
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nmslib
import re


def get_nms_query_method(data, k=1000,
                         index_time_params={'M': 15, 'indexThreadQty': 16, 'efConstruction': 200, 'post': 0, 'delaunay_type': 1}):
    query_time_params = {'efSearch': k}
    nms_index = nmslib.init(method='hnsw', space='cosinesimil')
    nms_index.addDataPointBatch(data)
    nms_index.createIndex(index_time_params, print_progress=True)
    nms_index.setQueryTimeParams(query_time_params)

    def query_method(v):
        neighbors, dist = nms_index.knnQuery(v, k=k)
        return dist, neighbors

    return query_method, nms_index


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def log_base_n(arr, base):
    return np.log(arr) / np.log(base)


def unit_length(a, axis=0):
    return a/np.expand_dims(norm(a, axis=axis), axis=axis)


def shuffle_copy(*args):
    rng_state = np.random.get_state()
    results = []
    for arg in args:
        res = np.copy(arg)
        np.random.shuffle(res)
        results.append(res)
        np.random.set_state(rng_state)
    return results[0] if len(args) == 1 else results


def average_precision_v1(y_true, y_pred):
    y_pred = np.array(y_pred)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 0]
    y_pred = np.array(y_pred).reshape((1, -1))[0]
    y_true = np.array(y_true).reshape((1, -1))[0]
    y_true = set(y_true)

    detections = [1 if y in y_true else 0 for y in y_pred]
    score = np.multiply(np.cumsum(detections), detections)
    divisors = np.arange(1, len(y_pred) + 1)
    score = np.divide(score, divisors)
    score = np.sum(score)
    return score / len(y_true)


def average_precision_v2(y_true, y_pred):
    y_pred = np.array(y_pred)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 0]
    y_pred = np.array(y_pred).reshape((1, -1))[0]
    y_true = np.array(y_true).reshape((1, -1))[0]
    y_true = set(y_true)

    score = 0.0
    matches_till_now = 0
    for i in range(len(y_pred)):
        if y_pred[i] in y_true:
            matches_till_now = matches_till_now + 1
            score = score + matches_till_now / (i + 1)

    return score / len(y_true)


def mean_average_precision(y_true: List[List[str]], y_pred: List[List[str]]):
    sn = []
    for i in range(len(y_true)):
        y = y_true[i]
        yp = y_pred[i]
        sn.append(average_precision(y, yp))
    return np.mean(sn)


def mean_average_precision_by_users(y_true: Dict[str, List[str]], y_pred: Dict[str, List[str]]):
    sn = []
    for k, v in y_true.items():
        yp = y_pred[k] if k in y_pred else []
        y = v
        sn.append(average_precision(y, yp))
    return np.mean(sn)

def ndcg(y_true: List[str], y_pred: List[str]):
    y_pred = np.array(y_pred)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 0]
    y_pred = np.array(y_pred).reshape((1, -1))[0]
    y_true = np.array(y_true).reshape((1, -1))[0]


def ndcg_by_users(y_true: Dict[str, List[str]], y_pred: Dict[str, List[str]]):
    sn = []
    for k, v in y_true.items():
        yp = y_pred[k] if k in y_pred else []
        y = v
        sn.append(ndcg(y, yp))
    return np.mean(sn)


def measure_array_dist_element_displacement(X1, X2):
    X1 = list(X1)
    X2 = list(X2)

    assert len(X1) == len(X2)
    diff = 0.
    elem_to_index = {e:i for i,e in enumerate(X1)}
    for index, element in enumerate(X2):
        actual_index = elem_to_index[element]
        diff += abs(index - actual_index)

    return diff / len(X1) ** 2 * 2


def measure_array_dist_inversions(X1, X2):
    def merge_sort_inv_counter(arr):
        return _merge_sort_counter(arr,[0]*len(arr),0,len(arr)-1)

    def _merge_sort_counter(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count = _merge_sort_counter(arr, temp_arr, left, mid)
            inv_count += _merge_sort_counter(arr, temp_arr, mid + 1, right)
            inv_count += merge(arr, temp_arr, left, mid, right)
        return inv_count

    def merge(arr, temp_arr, left, mid, right):
        i = left  # Starting index of left subarray
        j = mid + 1  # Starting index of right subarray
        k = left  # Starting index of to be sorted subarray
        inv_count = 0
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                k += 1
                i += 1
            else:
                # Inversion will occur.
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                k += 1
                j += 1

        while i <= mid:
            temp_arr[k] = arr[i]
            k += 1
            i += 1

        while j <= right:
            temp_arr[k] = arr[j]
            k += 1
            j += 1

        for loop_var in range(left, right + 1):
            arr[loop_var] = temp_arr[loop_var]

        return inv_count

    X1 = list(X1)
    X2 = list(X2)
    assert len(X1) == len(X2)
    elem_to_index_unsorted= {e: i for i, e in enumerate(X2)}
    unsorted = [elem_to_index_unsorted[e] for e in X1]
    return merge_sort_inv_counter(unsorted)/comb(len(X1),2)


def compare_embedding_global_distance_mismatches(high_dim_embeddigs, low_dim_embeddings, n_point_pairs=5):
    assert high_dim_embeddigs.shape[0] == low_dim_embeddings.shape[0]
    shuf_ind = np.array([],dtype=int)
    for i in range(n_point_pairs):
        shuf_ind = np.concatenate((shuf_ind,np.array(shuffle(list(range(len(high_dim_embeddigs)))))))

    def point_pair_distances(X, order):
        distances = np.sum(np.square(X[list(reversed(order))] - X[order]), axis=1)
        return list(distances)

    def get_sorted_distances(X):
        distances = point_pair_distances(X, shuf_ind)
        sd, si = zip(*list(sorted(zip(distances, range(len(distances))), key=lambda x:x[0])))
        si = np.array(si)
        return si
    sd1 = get_sorted_distances(high_dim_embeddigs)
    sd2 = get_sorted_distances(low_dim_embeddings)
    score = measure_array_dist_element_displacement(sd1, sd2)
    score2 = measure_array_dist_inversions(sd1, sd2)
    return score, score2


def auto_encoder_transform(Inputs, Outputs, n_dims=32,verbose=1,epochs=25 ):
    loss = "mean_squared_error"
    initial_dims = Inputs.shape[1]
    avg_value = 1.0/np.sqrt(initial_dims)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, verbose=0, )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=0.0001)
    input_layer = tf.keras.Input(shape=(Inputs.shape[1],))
    # encoded = tf.keras.layers.GaussianNoise(0.01)(input_layer)
    encoded = tf.keras.layers.Dense(n_dims * 4, activation='elu')(input_layer)
    encoded = tf.keras.layers.GaussianNoise(0.01*avg_value)(encoded)
    encoded = tf.keras.layers.Dense(n_dims * 2, activation='elu')(encoded)
    # encoded = tf.keras.layers.GaussianNoise(0.01)(encoded)
    encoded = tf.keras.layers.Dense(n_dims, activation='elu')(encoded)

    decoded = tf.keras.layers.Dense(n_dims * 2, activation='elu')(encoded)
    decoded = tf.keras.layers.GaussianNoise(0.01*avg_value)(decoded)
    decoded = tf.keras.layers.Dense(n_dims * 4, activation='elu')(decoded)
    decoded = tf.keras.layers.Dense(Outputs.shape[1], activation='elu')(decoded)

    autoencoder = tf.keras.Model(input_layer, decoded)
    encoder = tf.keras.Model(input_layer, encoded)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.05, amsgrad=False)
    autoencoder.compile(optimizer=adam, loss=loss)
    X1,X2, Y1,Y2 = train_test_split(Inputs, Outputs, test_size=0.5)
    autoencoder.fit(X1, Y1,
                    epochs=epochs,
                    batch_size=4096,
                    shuffle=True,
                    verbose=verbose,
                    validation_data=(X2, Y2),
                    callbacks=[es])

    autoencoder.fit(X2, Y2,
                    epochs=epochs,
                    batch_size=4096,
                    shuffle=True,
                    verbose=verbose,
                    validation_data=(X1, Y1),
                    callbacks=[es, reduce_lr])

    Z = encoder.predict(Inputs)
    return Z, encoder


def clean_text(text):
    EMPTY = ' '
    assert text is not None
    assert type(text) == str
    text = text.replace("'", " ").replace('"', " ")
    text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("\r", " ").replace("\t", " ").lower()
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    text = re.sub('<.*?>', EMPTY, text)
    return text


def repeat_args_wrapper(func):
    def wrapper(*args, **kwargs):
        results = []
        for arg in args:
            results.append(func(arg, **kwargs))
        return results[0] if len(results) == 1 else results
    return wrapper


def build_user_item_dict(user_item_affinities: List[Tuple[str, str, float]]):
    user_item_dict: Dict[str, Dict[str, float]] = {}

    for user, item, affinity in user_item_affinities:
        if user not in user_item_dict:
            user_item_dict[user] = {}
        user_item_dict[user][item] = affinity
    return user_item_dict


def build_item_user_dict(user_item_affinities: List[Tuple[str, str, float]]):
    item_user_dict: Dict[str, Dict[str, float]] = {}
    for user, item, affinity in user_item_affinities:
        if item not in item_user_dict:
            item_user_dict[item] = {}
        item_user_dict[item][user] = affinity
    return item_user_dict


def get_mean_rating(user_item_affinities: List[Tuple[str, str, float]]) -> float:
    uid = pd.DataFrame(user_item_affinities, columns=["user", "item", "rating"])
    mean = uid.rating.mean()
    return mean


def normalize_affinity_scores_by_user(user_item_affinities: List[Tuple[str, str, float]],) \
        -> Tuple[float, Dict[str, float], Dict[str, float], float, List[Tuple[str, str, float]]]:
    uid = pd.DataFrame(user_item_affinities, columns=["user", "item", "rating"])
    # Calculating Biases
    mean = uid.rating.mean()
    uid["rating"] = uid["rating"] - mean
    bu = uid.groupby(['user']).agg(['mean', 'min', 'max'])
    bu.columns = bu.columns.get_level_values(1)
    bu["spread"] = np.max((bu["max"] - bu["mean"], bu["mean"] - bu["min"]), axis=0)
    bu = bu.reset_index()
    uid = uid.merge(bu[["user", "mean"]], on="user")
    uid["rating"] = uid["rating"] - uid["mean"]

    bu = dict(zip(bu['user'], bu['mean']))
    # Stochastic Gradient Descent Taken from Surprise Lib
    lr = 0.001
    reg = 0.01
    n_epochs = 20
    for dummy in range(n_epochs):
        for u, i, r in user_item_affinities:
            err = (r - (mean + bu[u]))
            bu[u] += lr * (err - reg * bu[u])

    uid = [[u, i, r - (mean + bu[u])] for u, i, r in user_item_affinities]
    uid = pd.DataFrame(uid, columns=["user", "item", "rating"])
    unique_items = uid["item"].unique()
    bi = dict(zip(unique_items, [0]*len(unique_items)))

    # Calculating Spreads
    spread = max(uid["rating"].max(), np.abs(uid["rating"].min()))

    # Making final Dict
    uid = list(zip(uid['user'], uid['item'], uid['rating']))

    return mean, bu, bi, spread, uid


def normalize_affinity_scores_by_user_item(user_item_affinities: List[Tuple[str, str, float]],) \
        -> Tuple[float, Dict[str, float], Dict[str, float], float, List[Tuple[str, str, float]]]:
    uid = pd.DataFrame(user_item_affinities, columns=["user", "item", "rating"])
    # Calculating Biases
    mean = uid.rating.mean()
    uid["rating"] = uid["rating"] - mean
    bu = uid.groupby(['user']).agg(['mean', 'min', 'max'])
    bu.columns = bu.columns.get_level_values(1)
    bu["spread"] = np.max((bu["max"] - bu["mean"], bu["mean"] - bu["min"]), axis=0)
    bu = bu.reset_index()
    uid = uid.merge(bu[["user", "mean"]], on="user")
    uid["rating"] = uid["rating"] - uid["mean"]
    uid = uid[["user", "item", "rating"]]

    bi = uid[["user", "item", "rating"]].groupby(['item']).agg(['mean', 'min', 'max'])
    bi.columns = bi.columns.get_level_values(1)
    bi["spread"] = np.max((bi["max"] - bi["mean"], bi["mean"] - bi["min"]), axis=0)
    bi = bi.reset_index()
    uid = uid.merge(bi[["item", "mean"]], on="item")
    uid["rating"] = uid["rating"] - uid["mean"]

    bu = dict(zip(bu['user'], bu['mean']))
    bi = dict(zip(bi['item'], bi['mean']))
    # Stochastic Gradient Descent Taken from Surprise Lib
    lr = 0.001
    reg = 0.01
    n_epochs = 20
    for dummy in range(n_epochs):
        for u, i, r in user_item_affinities:
            err = (r - (mean + bu[u] + bi[i]))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

    uid = [[u, i, r - (mean + bu[u] + bi[i])] for u, i, r in user_item_affinities]
    uid = pd.DataFrame(uid, columns=["user", "item", "rating"])

    # Calculating Spreads
    spread = max(uid["rating"].max(), np.abs(uid["rating"].min()))

    # Making final Dict
    uid = list(zip(uid['user'], uid['item'], uid['rating']))

    return mean, bu, bi, spread, uid


def is_num(x):
    ans = isinstance(x, int) or isinstance(x, float) or isinstance(x, np.float)
    return ans


def is_1d_array(x):
    ans = (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, tuple)) \
          and not isinstance(x[0], list) and not isinstance(x[0], np.ndarray) and not isinstance(x[0], tuple)
    return ans


def is_2d_array(x):
    ans = (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, tuple)) and is_1d_array(x[0])
    return ans


unit_length = repeat_args_wrapper(unit_length)
average_precision = average_precision_v2

