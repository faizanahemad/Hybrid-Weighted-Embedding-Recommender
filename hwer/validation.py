
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from hwer.utils import average_precision

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
import os
import copy
from collections import defaultdict
import operator

warnings.filterwarnings('ignore')
from typing import List, Dict, Any, Tuple
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
from surprise import SVD, SVDpp
from surprise import accuracy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from ast import literal_eval

from .utils import normalize_affinity_scores_by_user



def surprise_get_topk(model, users, items, k=100) -> Dict[str, List[Tuple[str, float]]]:
    predictions = defaultdict(list)
    for u in users:
        p = [(i, model.predict(u, i).est) for i in items]
        p = list(sorted(p, key=operator.itemgetter(1), reverse=True))
        predictions[u] = p[:k]
    return predictions


def model_get_topk(model, users, items, k=100) -> Dict[str, List[Tuple[str, float]]]:
    predictions = defaultdict(list)
    for u in users:
        p = model.find_items_for_user(u)
        predictions[u] = p[:k]
    return predictions


def extraction_efficiency(model, train_affinities, validation_affinities, get_topk, item_list):
    validation_users = list(set([u for u, i, r in validation_affinities]))
    train_uid = defaultdict(set)
    items_extracted_length = []
    s = time.time()
    predictions = get_topk(model, validation_users, item_list)
    e = time.time()
    pred_time = e - s
    for u, i, r in train_affinities:
        train_uid[u].add(i)
    mean, bu, bi, _, _ = normalize_affinity_scores_by_user(train_affinities)
    for u, i in predictions.items():
        base_rating = mean + bu[u]
        remaining_items = list(sorted(filter(lambda x: x[1] >= base_rating, i), key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        predictions[u] = remaining_items

    validation_actuals = defaultdict(list)
    for u, i, r in validation_affinities:
        validation_actuals[u].append((i, r))

    for u, i in validation_actuals.items():
        base_rating = mean + bu[u]
        remaining_items = list(
            sorted(filter(lambda x: x[1] >= base_rating, i), key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        validation_actuals[u] = remaining_items

    mean_ap = np.mean([average_precision(validation_actuals[u], predictions[u]) for u in validation_users])

    # calculate ndcg
    return {"map": mean_ap, "retrieval_time": pred_time, "ndcg": 0.0}


def test_surprise(train, test, items, algo=["baseline", "svd", "svdpp"], algo_params={}, rating_scale=(1, 5)):
    train_affinities = train
    validation_affinities = test
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    reader = Reader(rating_scale=rating_scale)
    trainset = Dataset.load_from_df(train, reader).build_full_trainset()
    testset = Dataset.load_from_df(test, reader).build_full_trainset().build_testset()
    trainset_for_testing = trainset.build_testset()

    def use_algo(algo, name):
        start = time.time()
        algo.fit(trainset)
        predictions = algo.test(testset)
        end = time.time()
        total_time = end - start
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        ex_ee = extraction_efficiency(algo, train_affinities, validation_affinities, surprise_get_topk, items)

        predictions = algo.test(trainset_for_testing)
        train_rmse = accuracy.rmse(predictions, verbose=False)
        train_mae = accuracy.mae(predictions, verbose=False)
        return {"algo": name, "rmse": rmse, "mae": mae, "map": ex_ee["map"], "retrieval_time": ex_ee["retrieval_time"],
                "train_rmse": train_rmse, "train_mae": train_mae, "time": total_time}

    algo_map = {"svd": SVD(**(algo_params["svd"] if "svd" in algo_params else {})),
                "svdpp": SVDpp(**(algo_params["svdpp"] if "svdpp" in algo_params else {})),
                "baseline": BaselineOnly(bsl_options={'method': 'sgd'})}
    results = list(map(lambda a: use_algo(algo_map[a], a), algo))
    return results


def display_results(results: List[Dict[str, Any]]):
    df = pd.DataFrame.from_records(results)
    df = df.groupby(['algo']).mean()
    df['time'] = df['time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    df['retrieval_time'] = df['retrieval_time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    print(df)


def get_prediction_details(recsys, train_affinities, validation_affinities, model_get_topk, items):
    def get_details(recsys, affinities):
        predictions = recsys.predict([(u, i) for u, i, r in affinities])
        actuals = np.array([r for u, i, r in affinities])
        rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
        mae = np.mean(np.abs(actuals - predictions))
        return predictions, actuals, rmse, mae
    _, _, train_rmse, train_mae = get_details(recsys, train_affinities)
    predictions, actuals, rmse, mae = get_details(recsys, validation_affinities)
    print(rmse, mae, train_rmse, train_mae)
    ex_ee = extraction_efficiency(recsys, train_affinities, validation_affinities, model_get_topk, items)
    stats = {"rmse": rmse, "mae": mae,
            "map": ex_ee["map"], "retrieval_time": ex_ee["retrieval_time"],
            "train_rmse": train_rmse, "train_mae": train_mae}
    return predictions, actuals, stats


def error_analysis(error_df, title):
    print("-x-" * 30)
    print("%s: Error Analysis -: " % title)

    print(error_df.describe())

    print("Analysis By actuals")
    print(error_df.groupby(["actuals"]).agg(["mean", "std"]))

    print("Describe Errors -: ")
    print(describe(error_df["errors"].values))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="actuals", y="errors", data=error_df)
    plt.title("Errors vs Actuals")
    plt.xlabel("Actuals")
    plt.ylabel("Errors")
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="predictions", y="errors", hue="actuals", data=error_df)
    plt.title("Errors vs Predictions")
    plt.xlabel("Predictions")
    plt.ylabel("Errors")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(error_df["errors"], bins=100)
    plt.title("Error Histogram")
    plt.show()
