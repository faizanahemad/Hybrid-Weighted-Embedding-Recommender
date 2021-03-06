import re
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from scipy.special import comb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse


def get_cosine_schedule_with_warmup(optimizer, epochs, batch_size, n_samples):
    from transformers import optimization
    warmup_proportion = 0.3
    n_steps = int(np.ceil(n_samples / batch_size))
    num_training_steps = n_steps * epochs
    num_warmup_steps = int(warmup_proportion * num_training_steps)
    sch = optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    return sch


def get_constant_schedule_with_warmup(optimizer, epochs, batch_size, n_samples):
    from transformers import optimization
    warmup_proportion = 0.3
    n_steps = int(np.ceil(n_samples / batch_size))
    num_training_steps = n_steps * epochs
    num_warmup_steps = int(warmup_proportion * num_training_steps)
    sch = optimization.get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    return sch


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def log_base_n(arr, base):
    return np.log(arr) / np.log(base)


def unit_length(a, axis=0):
    return a / np.expand_dims(norm(a, axis=axis), axis=axis)


def get_nan_rows(a, axis=1):
    return np.sum(np.sum(np.isnan(a), axis=axis) > 0)


def unit_length_violations(a, axis=0, epsilon=1e-4):
    vector_lengths = np.expand_dims(norm(a, axis=axis), axis=axis)
    positive_violations = np.sum(vector_lengths > 1 + epsilon)
    negative_violations = np.sum(vector_lengths < 1 - epsilon)
    violations = positive_violations + negative_violations
    violation_mean = np.mean(np.abs(vector_lengths - 1))
    return violations, violation_mean, positive_violations, negative_violations


def shuffle_copy(*args):
    rng_state = np.random.get_state()
    results = []
    for arg in args:
        res = np.copy(arg)
        np.random.shuffle(res)
        results.append(res)
        np.random.set_state(rng_state)
    return results[0] if len(args) == 1 else results


def reciprocal_rank(y_true, y_pred):
    y_true = set(y_true)
    rr = 0.0
    for i, e in enumerate(y_pred):
        if e in y_true:
            rr = 1.0/(i+1)
            return rr
    return rr


def average_precision(y_true, y_pred):
    len_y_true = max(1, len(y_true))
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
            y_true.discard(y_pred[i])

    return score / len_y_true


def ndcg(y_true: Dict[str, float], y_pred: List[str]):
    y_true_sorted = sorted(y_true.values(), reverse=True)
    y_true_sorted = y_true_sorted[:len(y_pred)]
    idcg = np.sum((np.power(2, y_true_sorted) - 1)/(np.log2(np.arange(len(y_true_sorted)) + 2)))
    y_pred = [y_true[i] if i in y_true else 0 for i in y_pred]
    dcg = np.sum((np.power(2, y_pred) - 1) / (np.log2(np.arange(len(y_pred)) + 2)))
    return dcg/(idcg + 1e-8)


def binary_ndcg(y_true: Dict[str, float], y_pred: List[str]):
    return ndcg({k:1 for k, v in y_true.items()}, y_pred)


def binary_ndcg_v2(y_true: List[str], y_pred: List[str]):
    return ndcg({k: 1 for k in y_true}, y_pred)


def recall(y_true: Dict[str, float], y_pred: List[str]):
    norm = min(len(y_pred), len(y_true))
    recall = sum([1 if i in y_true else 0 for i in y_pred])
    return recall/max(norm, 1.0)


def measure_array_dist_element_displacement(X1, X2):
    X1 = list(X1)
    X2 = list(X2)

    assert len(X1) == len(X2)
    diff = 0.
    elem_to_index = {e: i for i, e in enumerate(X1)}
    for index, element in enumerate(X2):
        actual_index = elem_to_index[element]
        diff += abs(index - actual_index)

    return diff / len(X1) ** 2 * 2


def measure_array_dist_inversions(X1, X2):
    def merge_sort_inv_counter(arr):
        return _merge_sort_counter(arr, [0] * len(arr), 0, len(arr) - 1)

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
    elem_to_index_unsorted = {e: i for i, e in enumerate(X2)}
    unsorted = [elem_to_index_unsorted[e] for e in X1]
    return merge_sort_inv_counter(unsorted) / comb(len(X1), 2)


def compare_embedding_global_distance_mismatches(high_dim_embeddigs, low_dim_embeddings, n_point_pairs=5):
    assert high_dim_embeddigs.shape[0] == low_dim_embeddings.shape[0]
    shuf_ind = np.array([], dtype=int)
    for i in range(n_point_pairs):
        shuf_ind = np.concatenate((shuf_ind, np.array(shuffle(list(range(len(high_dim_embeddigs)))))))

    def point_pair_distances(X, order):
        distances = np.sum(np.square(X[list(reversed(order))] - X[order]), axis=1)
        return list(distances)

    def get_sorted_distances(X):
        distances = point_pair_distances(X, shuf_ind)
        sd, si = zip(*list(sorted(zip(distances, range(len(distances))), key=lambda x: x[0])))
        si = np.array(si)
        return si

    sd1 = get_sorted_distances(high_dim_embeddigs)
    sd2 = get_sorted_distances(low_dim_embeddings)
    score = measure_array_dist_element_displacement(sd1, sd2)
    score2 = measure_array_dist_inversions(sd1, sd2)
    return score, score2


def auto_encoder_transform(Inputs, Outputs, n_dims=32, verbose=0, epochs=10):
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow import keras
    loss = "mean_squared_error"
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, verbose=0, )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.0001)
    nan_prevent = tf.keras.callbacks.TerminateOnNaN()
    input_layer = tf.keras.Input(shape=(Inputs.shape[1],))
    encoded = tf.keras.layers.Dense(n_dims * 4, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l2=1e-5), use_bias=False)(input_layer)
    encoded = tf.keras.layers.Dense(n_dims, activation='linear', use_bias=False)(encoded)
    encoded = K.l2_normalize(encoded, axis=-1)

    decoded = tf.keras.layers.Dense(n_dims * 4, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l2=1e-5), use_bias=False)(encoded)
    decoded = tf.keras.layers.Dense(Outputs.shape[1], use_bias=False)(decoded)

    autoencoder = tf.keras.Model(input_layer, decoded)
    encoder = tf.keras.Model(input_layer, encoded)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer=adam, loss=loss, metrics=["mean_squared_error"])
    Inputs = Inputs.astype(float)
    Outputs = Outputs.astype(float)
    X1, X2, Y1, Y2 = train_test_split(Inputs, Outputs, test_size=0.5)
    autoencoder.fit(X1, Y1,
                    epochs=epochs,
                    batch_size=2048,
                    shuffle=True,
                    verbose=verbose,
                    validation_data=(X2, Y2),
                    callbacks=[es, reduce_lr, nan_prevent])

    K.set_value(autoencoder.optimizer.lr, 0.001)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, verbose=0, )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=0.00001)
    autoencoder.fit(X2, Y2,
                    epochs=epochs,
                    batch_size=2048,
                    shuffle=True,
                    verbose=verbose,
                    validation_data=(X1, Y1),
                    callbacks=[es, reduce_lr, nan_prevent])

    K.set_value(autoencoder.optimizer.lr, 0.001)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.2, patience=1, min_lr=0.00001)
    autoencoder.fit(Inputs, Outputs,
                    epochs=epochs,
                    batch_size=2048,
                    shuffle=True,
                    verbose=verbose,
                    callbacks=[reduce_lr, nan_prevent])

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


def is_num(x):
    ans = isinstance(x, int) or isinstance(x, float) or isinstance(x, np.float) or isinstance(x, np.int) or isinstance(x, np.int64)
    return ans


def is_1d_array(x):
    ans = (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, tuple)) \
          and not isinstance(x[0], list) and not isinstance(x[0], np.ndarray) and not isinstance(x[0], tuple)
    return ans


def is_2d_array(x):
    ans = (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, tuple)) and is_1d_array(x[0])
    return ans


unit_length = repeat_args_wrapper(unit_length)
unit_length_violations = repeat_args_wrapper(unit_length_violations)


def get_rng(noise_augmentation):
    if noise_augmentation:
        def rng(dims):
            r = np.random.rand(dims) if dims > 1 else np.random.rand() - 0.5
            return noise_augmentation * r

        return rng
    return lambda dims: np.zeros(dims) if dims > 1 else 0


class NodeNotFoundException(Exception):
    pass


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_list_per_line(lines, filename, mode='a'):
    # convert lines to a single blob of text
    lines = list(map(lambda s: " ".join(list(map(lambda x: str(x).strip(), s))), lines))
    data = '\n'.join(lines)
    with open(filename, mode) as file:
        file.write(data)


def merge_dicts_nested(d1: Dict[object, Dict], d2: Dict[object, Dict], *args):
    from more_itertools import flatten
    assert set(d1.keys()) == set(d2.keys())
    other_dict_lens = len(args)
    if other_dict_lens > 0:
        assert set(d1.keys()) == set(list(flatten([set(d.keys()) for d in args])))
    for k, v in d1.items():
        v.update(d2[k])
        for i in range(other_dict_lens):
            v.update(args[i][k])
    return d1


def build_row_dicts(keyword, rows):
    return [{keyword: row} for row in rows]

