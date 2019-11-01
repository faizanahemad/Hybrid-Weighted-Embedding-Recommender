from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import nmslib
import time
from tqdm import tqdm_notebook
from fasttext import FastText


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def log_base_n(arr, base):
    return np.log(arr) / np.log(base)


def unit_length(a, axis=0):
    return a/np.expand_dims(norm(a, axis=axis), axis=axis)


def fasttext_get_sentence_vectorizer(fasttext_model:FastText):
    def get_sentence_vector(text):
        v = fasttext_model.get_sentence_vector(text)
        v = unit_length(v)
        return v
    return get_sentence_vector


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


def mean_average_precision(y_true, y_pred, n):
    sn = []
    for i in range(len(y_true)):
        y = y_true[i]
        yp = y_pred[i]
        sn.append(average_precision(y, yp, n))
    return np.mean(sn)


average_precision = average_precision_v2

