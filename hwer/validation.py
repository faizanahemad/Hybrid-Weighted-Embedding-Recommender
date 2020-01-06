
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from surprise.prediction_algorithms.co_clustering import CoClustering

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
from surprise import SVD, SVDpp, NormalPredictor
from surprise import accuracy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from ast import literal_eval

from .utils import normalize_affinity_scores_by_user, average_precision
from . import SVDppHybrid, ContentRecommendation, HybridGCNRec


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


def extraction_efficiency(model, train_affinities, validation_affinities, get_topk, item_list, min_positive_rating=None):
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
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        predictions[u] = remaining_items

    validation_actuals = defaultdict(list)
    for u, i, r in validation_affinities:
        validation_actuals[u].append((i, r))

    for u, i in validation_actuals.items():
        br = (mean + bu[u]) if min_positive_rating is None else min_positive_rating
        remaining_items = list(
            sorted(filter(lambda x: x[1] >= br, i), key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        validation_actuals[u] = remaining_items

    mean_ap = np.mean([average_precision(validation_actuals[u], predictions[u]) for u in validation_users])

    return {"map": mean_ap, "retrieval_time": pred_time, "ndcg": 0.0,
            "actuals": validation_actuals, "predictions": predictions}


def test_surprise(train, test, algo=["baseline", "svd", "svdpp"], algo_params={}, rating_scale=(1, 5),
                  ignore_below_rating=None, min_positive_rating=None):
    train_affinities = train
    validation_affinities = test
    items = list(set([i for u, i, r in train]))
    train = pd.DataFrame(train)
    test = pd.DataFrame([(u, i, r) for u, i, r in test if r >= ignore_below_rating])
    reader = Reader(rating_scale=rating_scale)
    trainset = Dataset.load_from_df(train, reader).build_full_trainset()
    testset = Dataset.load_from_df(test, reader).build_full_trainset().build_testset()
    trainset_for_testing = Dataset.load_from_df(pd.DataFrame([(u, i, r) for u, i, r in train_affinities if r >= ignore_below_rating]), reader).build_full_trainset().build_testset()

    def use_algo(algo, name):
        start = time.time()
        algo.fit(trainset)
        predictions = algo.test(testset)
        end = time.time()
        total_time = end - start
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        if ignore_below_rating is not None:
            train_aff = [(u, i, r) for u, i, r in train_affinities if r >= ignore_below_rating]
            validation_aff = [(u, i, r) for u, i, r in validation_affinities if r >= ignore_below_rating]

        ex_ee = extraction_efficiency(algo, train_aff, validation_aff, surprise_get_topk, items, min_positive_rating=min_positive_rating)

        train_predictions = algo.test(trainset_for_testing)
        train_rmse = accuracy.rmse(train_predictions, verbose=False)
        train_mae = accuracy.mae(train_predictions, verbose=False)
        train_predictions = [p.est for p in train_predictions]
        predictions = [p.est for p in predictions]
        user_rating_count_metrics = metrics_by_num_interactions_user(train_aff, validation_aff,
                                                                     train_predictions, predictions,
                                                                     ex_ee["actuals"], ex_ee["predictions"])
        user_rating_count_metrics["algo"] = name
        stats = {"algo": name, "rmse": rmse, "mae": mae, "map": ex_ee["map"], "retrieval_time": ex_ee["retrieval_time"],
                "train_rmse": train_rmse, "train_mae": train_mae, "time": total_time,
                 "user_rating_count_metrics": user_rating_count_metrics}
        return stats

    algo_map = {"svd": SVD(**(algo_params["svd"] if "svd" in algo_params else {})),
                "svdpp": SVDpp(**(algo_params["svdpp"] if "svdpp" in algo_params else {})),
                "baseline": BaselineOnly(bsl_options={'method': 'sgd'}),
                "clustering": CoClustering(**(algo_params["clustering"] if "clustering" in algo_params else dict(n_cltr_u=5, n_cltr_i=10))),
                "normal": NormalPredictor()}
    results = list(map(lambda a: use_algo(algo_map[a], a), algo))
    user_rating_count_metrics = pd.concat([r["user_rating_count_metrics"] for r in results])
    for s in results:
        del s["user_rating_count_metrics"]
    return None, results, user_rating_count_metrics, None, None


def test_hybrid(train_affinities, validation_affinities, users, items, hyperparameters,
                      get_data_mappers, rating_scale,
                min_positive_rating, ignore_below_rating, algo,
                      enable_error_analysis=False):
    embedding_mapper, user_data, item_data = get_data_mappers()
    kwargs = dict(user_data=user_data, item_data=item_data, hyperparameters=copy.deepcopy(hyperparameters))
    if algo == "svdpp_hybrid":
        recsys = SVDppHybrid(embedding_mapper=embedding_mapper,
                             knn_params=hyperparameters["knn_params"],
                             rating_scale=rating_scale,
                             n_content_dims=hyperparameters["n_dims"],
                             n_collaborative_dims=hyperparameters["n_dims"],
                             fast_inference=False, super_fast_inference=False)
    elif algo == "gcn_hybrid":
        recsys = HybridGCNRec(embedding_mapper=embedding_mapper,
                              knn_params=hyperparameters["knn_params"],
                              rating_scale=rating_scale,
                              n_content_dims=hyperparameters["n_dims"],
                              n_collaborative_dims=hyperparameters["n_dims"],
                              fast_inference=False, super_fast_inference=False)

    start = time.time()
    _, _ = recsys.fit(users, items,
                      train_affinities, **kwargs)
    end = time.time()
    total_time = end - start

    assert np.sum(np.isnan(recsys.predict([(users[0], "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl"),
                                           (users[0], items[0]),
                                           ("21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl", "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl"),
                                           ("21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl", items[0])]))) == 0

    recsys.fast_inference = True
    recsys.super_fast_inference = False
    res2 = {"algo": "Fast-%s" % algo, "time": total_time}
    predictions, actuals, stats, user_rating_count_metrics = get_prediction_details(recsys, train_affinities,
                                                                                    validation_affinities,
                                                                                    model_get_topk, items,
                                                                                    min_positive_rating=min_positive_rating,
                                                                                    ignore_below_rating=ignore_below_rating)
    res2.update(stats)
    user_rating_count_metrics["algo"] = res2["algo"]

    #
    recsys.fast_inference = False
    recsys.super_fast_inference = True
    res4 = {"algo": "Super-Fast-%s" % algo, "time": total_time}
    predictions, actuals, stats, urcm = get_prediction_details(recsys, train_affinities,
                                                                                    validation_affinities,
                                                                                    model_get_topk, items,
                                                                                    min_positive_rating=min_positive_rating,
                                                                                    ignore_below_rating=ignore_below_rating)
    res4.update(stats)
    urcm["algo"] = res4["algo"]
    user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))

    #
    recsys.fast_inference = False
    recsys.super_fast_inference = False
    res = {"algo": algo, "time": total_time}
    predictions, actuals, stats, urcm = get_prediction_details(recsys, train_affinities, validation_affinities,
                                                               model_get_topk, items, min_positive_rating=min_positive_rating,
                                                               ignore_below_rating=ignore_below_rating)
    res.update(stats)
    urcm["algo"] = res["algo"]
    user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(train_affinities, validation_affinities, error_df, "Hybrid")
    results = [res, res2, res4]
    return recsys, results, user_rating_count_metrics, predictions, actuals


def test_content_only(train_affinities, validation_affinities, users, items, hyperparameters,
                      get_data_mappers, rating_scale, min_positive_rating, ignore_below_rating,
                      enable_error_analysis=False):
    embedding_mapper, user_data, item_data = get_data_mappers()
    kwargs = dict(user_data=user_data, item_data=item_data, hyperparameters=copy.deepcopy(hyperparameters))
    recsys = ContentRecommendation(embedding_mapper=embedding_mapper,
                                           knn_params=hyperparameters["knn_params"],
                                           rating_scale=rating_scale, n_output_dims=hyperparameters["n_dims"], )
    start = time.time()
    _, _ = recsys.fit(users, items,
                              train_affinities, **kwargs)
    end = time.time()
    total_time = end - start
    assert np.sum(np.isnan(recsys.predict([(users[0], "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl")]))) == 0

    res = {"algo": "Content-Only", "time": total_time}
    predictions, actuals, stats, user_rating_count_metrics = get_prediction_details(recsys, train_affinities,
                                                                                    validation_affinities,
                                                                                    model_get_topk, items,
                                                                                    min_positive_rating=min_positive_rating,
                                                                                    ignore_below_rating=ignore_below_rating)
    res.update(stats)
    user_rating_count_metrics["algo"] = res["algo"]

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(train_affinities, validation_affinities, error_df, "Hybrid")
    results = [res]
    return recsys, results, user_rating_count_metrics, predictions, actuals


def test_once(train_affinities, validation_affinities, users, items, hyperparamters_dict,
              get_data_mappers, rating_scale, min_positive_rating, ignore_below_rating,
              svdpp_hybrid=True, gcn_hybrid=True, surprise=True, content_only=True, enable_error_analysis=False):
    results = []
    recs = []
    user_rating_count_metrics = pd.DataFrame([], columns=["algo", "user_rating_count", "rmse", "mae", "map", "train_rmse", "train_mae"])
    if surprise:
        hyperparameters_surprise = hyperparamters_dict["surprise"]
        _, surprise_results, surprise_user_rating_count_metrics, _, _ = test_surprise(train_affinities,
                                                                                      validation_affinities,
                                                                                      algo=hyperparameters_surprise[
                                                                                          "algos"],
                                                                                      algo_params=hyperparameters_surprise,
                                                                                      rating_scale=rating_scale,
                                                                                      min_positive_rating=min_positive_rating,
                                                                                      ignore_below_rating=ignore_below_rating)
        results.extend(surprise_results)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, surprise_user_rating_count_metrics))

    if content_only:
        hyperparameters = hyperparamters_dict["content_only"]
        content_rec, res, content_user_rating_count_metrics, _, _ = test_content_only(train_affinities, validation_affinities, users,
                                                                                      items, hyperparameters, get_data_mappers, rating_scale,
                                                                                      min_positive_rating, ignore_below_rating,
                                                                                      enable_error_analysis=enable_error_analysis)
        results.extend(res)
        recs.append(content_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, content_user_rating_count_metrics))

    if svdpp_hybrid:
        hyperparameters = hyperparamters_dict["svdpp_hybrid"]
        svd_rec, res, svdpp_user_rating_count_metrics, _, _ = test_hybrid(train_affinities, validation_affinities, users,
                                                                              items, hyperparameters, get_data_mappers, rating_scale,
                                                                          min_positive_rating, ignore_below_rating,
                                                                          algo="svdpp_hybrid",
                                                                                enable_error_analysis=enable_error_analysis)
        results.extend(res)
        recs.append(svd_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, svdpp_user_rating_count_metrics))

    if gcn_hybrid:
        hyperparameters = hyperparamters_dict["gcn_hybrid"]
        gcn_rec, res, gcn_user_rating_count_metrics, _, _ = test_hybrid(train_affinities, validation_affinities, users,
                                                                          items, hyperparameters, get_data_mappers, rating_scale,
                                                                            min_positive_rating, ignore_below_rating,
                                                                            algo="gcn_hybrid",
                                                                            enable_error_analysis=enable_error_analysis)
        results.extend(res)
        recs.append(gcn_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, gcn_user_rating_count_metrics))
    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    return recs, results, user_rating_count_metrics


def metrics_by_num_interactions_user(train_affinities: List[Tuple[str, str, float]], validation_affinities: List[Tuple[str, str, float]],
                                     train_predictions: np.ndarray, val_predictions: np.ndarray,
                                     val_user_topk_actuals: Dict[str, List[str]], val_user_topk_predictions: Dict[str, List[str]],
                                     mode="le",
                                     increments=2):
    columns = ["user", "item", "rating"]
    train_affinities = pd.DataFrame(train_affinities, columns=columns)
    validation_affinities = pd.DataFrame(validation_affinities, columns=columns)
    train_affinities["prediction"] = train_predictions
    validation_affinities["prediction"] = val_predictions
    user_rating_count = train_affinities.groupby(["user"])["item"].agg(['count']).reset_index()
    min_count = user_rating_count["count"].min()
    max_count = user_rating_count["count"].max()

    def rmse_mae_calc(affinities):
        error = affinities["rating"] - affinities["prediction"]
        rmse = np.sqrt(np.mean(np.square(error)))
        mae = np.mean(np.abs(error))
        return rmse, mae

    results = []
    for i in range(min_count, max_count, increments):
        uc = min(i, max_count)
        if mode == "exact":
            users = set(user_rating_count[user_rating_count["count"] == uc]['user'])
        elif mode == "le":
            users = set(user_rating_count[user_rating_count["count"] <= uc]['user'])
        elif mode == "ge":
            users = set(user_rating_count[user_rating_count["count"] >= uc]['user'])
        else:
            raise ValueError("`mode` should be one of {'exact', 'le', 'ge'}")
        train_rmse, train_mae = rmse_mae_calc(train_affinities[train_affinities.user.isin(users)])
        val_rmse, val_mae = rmse_mae_calc(validation_affinities[validation_affinities.user.isin(users)])
        mean_ap = np.mean([average_precision(val_user_topk_actuals[u], val_user_topk_predictions[u]) for u in users])

        results.append(["", uc, val_rmse, val_mae, mean_ap, train_rmse, train_mae])
    results = pd.DataFrame(results, columns=["algo", "user_rating_count", "rmse", "mae", "map", "train_rmse", "train_mae"])
    return results


def display_results(results: List[Dict[str, Any]]):
    df = pd.DataFrame.from_records(results)
    df = df.groupby(['algo']).mean()
    df['time'] = df['time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    time = df['retrieval_time']
    df['retrieval_time'] = df['retrieval_time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    print(df)
    df['retrieval_time'] = time
    return df


def visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities):
    # TODO: combining factor as X-axis, y-axis as map, hue as user_rating_count
    validation_users_count = len(set([u for u, i, r in validation_affinities]))
    results['retrieval_time'] = results['retrieval_time'] / validation_users_count
    results['retrieval_time'] = results['retrieval_time'] * 1000
    plt.figure(figsize=(12, 8))
    plt.title("Retrieval Time vs Algorithm")
    sns.barplot(results.index, results.retrieval_time)
    plt.xlabel("Algorithm")
    plt.ylabel("Retrieval Time in milli-seconds")
    plt.xticks(rotation=45, ha='right')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Mean Absolute Error vs Algorithm")
    sns.barplot(results.index, results.mae)
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha='right')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Mean Average Precision vs Algorithm")
    sns.barplot(results.index, results.map)
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Average Precision")
    plt.xticks(rotation=45, ha='right')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Mean Absolute Error vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="mae", hue="algo", style="algo", markers=True, data=user_rating_count_metrics)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Mean Average Precision vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="map", hue="algo", style="algo", markers=True, data=user_rating_count_metrics)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def get_prediction_details(recsys, train_affinities, validation_affinities, model_get_topk, items,
                           ignore_below_rating=None, min_positive_rating=None):
    if ignore_below_rating is not None:
        train_affinities = [(u, i, r) for u, i, r in train_affinities if r >= ignore_below_rating]
        validation_affinities = [(u, i, r) for u, i, r in validation_affinities if r >= ignore_below_rating]

    def get_details(recsys, affinities):
        predictions = np.array(recsys.predict([(u, i) for u, i, r in affinities]))
        assert np.sum(np.isnan(predictions)) == 0
        assert np.sum(predictions <= 0) == 0
        actuals = np.array([r for u, i, r in affinities])
        rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
        mae = np.mean(np.abs(actuals - predictions))
        return predictions, actuals, rmse, mae
    predictions, actuals, rmse, mae = get_details(recsys, validation_affinities)
    train_predictions, _, train_rmse, train_mae = get_details(recsys, train_affinities)
    ex_ee = extraction_efficiency(recsys, train_affinities, validation_affinities, model_get_topk, items, min_positive_rating=min_positive_rating)
    user_rating_count_metrics = metrics_by_num_interactions_user(train_affinities, validation_affinities, train_predictions, predictions,
                                                                 ex_ee["actuals"], ex_ee["predictions"])
    stats = {"rmse": rmse, "mae": mae,
            "map": ex_ee["map"], "retrieval_time": ex_ee["retrieval_time"],
            "train_rmse": train_rmse, "train_mae": train_mae}
    return predictions, actuals, stats, user_rating_count_metrics


def error_analysis(train_affinities, validation_affinities, error_df, title):
    # TODO: Error vs User Rating Count
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
