import pandas as pd
from bidict import bidict
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
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

from .utils import average_precision, reciprocal_rank, ndcg, binary_ndcg, recall, binary_ndcg_v2


def surprise_get_topk(model, users, items) -> Dict[str, List[Tuple[str, float]]]:
    predictions = defaultdict(list)
    for u in users:
        p = [(i, model.predict(u, i).est) for i in items]
        p = list(sorted(p, key=operator.itemgetter(1), reverse=True))
        predictions[u] = p
    return predictions


def model_get_topk_knn(model, users, items) -> Dict[str, List[Tuple[str, float]]]:
    predictions = defaultdict(list)
    for u in users:
        p = model.find_items_for_user(u)
        predictions[u] = p
    return predictions


def model_get_all(model, users, items) -> Dict[str, List[Tuple[str, float]]]:
    predictions = defaultdict(list)
    for u in users:
        p = list(zip(items, model.predict([(u, i) for i in items])))
        p = list(sorted(p, key=operator.itemgetter(1), reverse=True))
        predictions[u] = p
    return predictions


model_get_topk = model_get_topk_knn


def ncf_eval(model, train_affinities, validation_affinities, get_topk, item_list):
    item_list = set(item_list)
    train_interactions = defaultdict(set)
    for u, i, _ in train_affinities:
        train_interactions[u].add(i)
    user_test_item = {}
    actual = {}
    for u, i, _ in validation_affinities:
        user_test_item[u] = [i, *random.sample(item_list - train_interactions[u], 100)]
        actual[u] = i

    for u, items in user_test_item.items():
        it = list(zip(items, model.predict([(u, i) for i in items])))
        it = list(sorted(it, key=operator.itemgetter(1), reverse=True))
        user_test_item[u], _ = zip(*it[:10])

    hr = []
    ndcg = []
    for u, i in actual.items():
        preds = user_test_item[u]
        hr.append(i in preds)
        ndcg.append(binary_ndcg_v2([i], preds))

    return {"ncf_hr": np.mean(hr), "ncf_ndcg": np.mean(ndcg)}


def extraction_efficiency(model, train_affinities, validation_affinities, get_topk, item_list):
    validation_users = list(set([u for u, i, r in validation_affinities]))
    train_users = list(set([u for u, i, r in train_affinities]))
    all_users = list(set(train_users + validation_users))
    train_uid = defaultdict(set)
    items_extracted_length = []
    s = time.time()
    predictions = get_topk(model, all_users, item_list)
    e = time.time()
    pred_time = e - s
    for u, i, r in train_affinities:
        train_uid[u].add(i)

    train_actuals = defaultdict(list)
    train_actuals_score_dict = defaultdict(dict)
    for u, i, r in train_affinities:
        train_actuals[u].append((i, r))
        train_actuals_score_dict[u][i] = r

    for u, i in train_actuals.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = [i for i, r in remaining_items]
        train_actuals[u] = remaining_items

    train_predictions = dict()
    predictions_10 = dict()
    predictions_20 = dict()
    predictions_50 = dict()
    predictions_100 = dict()
    for u, i in predictions.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = [i for i, r in remaining_items]
        filtered_items = list(filter(lambda x: x not in train_uid[u], remaining_items))
        train_predictions[u] = remaining_items[:100]
        predictions_10[u] = filtered_items[:10]
        predictions_20[u] = filtered_items[:20]
        predictions_50[u] = filtered_items[:50]
        predictions_100[u] = filtered_items[:100]

    from more_itertools import flatten
    train_diversity = len(set(list(flatten(list(train_predictions.values())))))/len(item_list)
    diversity = len(set(list(flatten(list(predictions_100.values()))))) / len(item_list)

    train_mean_ap = np.mean([average_precision(train_actuals[u], train_predictions[u]) for u in train_users])
    train_mrr = np.mean([reciprocal_rank(train_actuals[u], train_predictions[u]) for u in train_users])
    train_ndcg = np.mean([ndcg(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])
    train_binary_ndcg = np.mean([binary_ndcg(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])
    train_recall = np.mean([recall(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])

    validation_actuals = defaultdict(list)
    for u, i, r in validation_affinities:
        validation_actuals[u].append((i, r))

    validation_actuals_score_dict = defaultdict(dict)
    for u, i in validation_actuals.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        validation_actuals_score_dict[u] = dict(remaining_items)
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        validation_actuals[u] = remaining_items

    mean_ap = np.mean([average_precision(validation_actuals[u], predictions_100[u]) for u in validation_users])
    mrr = np.mean([reciprocal_rank(validation_actuals[u], predictions_100[u]) for u in validation_users])
    val_ndcg = np.mean([ndcg(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])
    val_binary_ndcg = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])
    val_recall = np.mean([recall(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])

    mean_ap_10 = np.mean([average_precision(validation_actuals[u], predictions_10[u]) for u in validation_users])
    mrr_10 = np.mean([reciprocal_rank(validation_actuals[u], predictions_10[u]) for u in validation_users])
    val_ndcg_10 = np.mean([ndcg(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])
    val_binary_ndcg_10 = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])
    val_recall_10 = np.mean([recall(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])

    mean_ap_20 = np.mean([average_precision(validation_actuals[u], predictions_20[u]) for u in validation_users])
    mrr_20 = np.mean([reciprocal_rank(validation_actuals[u], predictions_20[u]) for u in validation_users])
    val_ndcg_20 = np.mean([ndcg(validation_actuals_score_dict[u], predictions_20[u]) for u in validation_users])
    val_binary_ndcg_20 = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_20[u]) for u in validation_users])
    val_recall_20 = np.mean([recall(validation_actuals_score_dict[u], predictions_20[u]) for u in validation_users])

    mean_ap_50 = np.mean([average_precision(validation_actuals[u], predictions_50[u]) for u in validation_users])
    mrr_50 = np.mean([reciprocal_rank(validation_actuals[u], predictions_50[u]) for u in validation_users])
    val_ndcg_50 = np.mean([ndcg(validation_actuals_score_dict[u], predictions_50[u]) for u in validation_users])
    val_binary_ndcg_50 = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_50[u]) for u in validation_users])
    val_recall_50 = np.mean([recall(validation_actuals_score_dict[u], predictions_50[u]) for u in validation_users])

    ncf_metrics = ncf_eval(model, train_affinities, validation_affinities, get_topk, item_list)

    metrics = {"train_map": train_mean_ap, "map": mean_ap, "train_mrr": train_mrr, "mrr": mrr,
               "retrieval_time": pred_time,
               "train_ndcg@100": train_ndcg, "ndcg@100": val_ndcg,
               "train_recall@100": train_recall, "recall@100": val_recall,
               "train_b_ndcg@100": train_binary_ndcg, "ndcg_b@100": val_binary_ndcg,
               "ndcg@50": val_ndcg_50, "ndcg_b@50": val_binary_ndcg_50,
               "ndcg@20": val_ndcg_20, "ndcg_b@20": val_binary_ndcg_20,
               "ndcg@10": val_ndcg_10, "ndcg_b@10": val_binary_ndcg_10,
               "recall@10": val_recall_10, "recall@20": val_recall_20, "recall@50": val_recall_50,
               "train_diversity": train_diversity, "diversity": diversity, **ncf_metrics}
    return {"actuals": validation_actuals, "predictions": predictions_100,
            "train_actuals": train_actuals, "train_predictions": train_predictions,
            "train_actuals_score_dict": train_actuals_score_dict,
            "validation_actuals_score_dict": validation_actuals_score_dict, "metrics": metrics}


def test_surprise(train, test, algo=("baseline", "svd", "svdpp"), algo_params={}, rating_scale=(1, 5)):
    from surprise import SVD, SVDpp, NormalPredictor
    from surprise import accuracy
    from surprise import BaselineOnly
    from surprise import Dataset
    from surprise import Reader
    from surprise.prediction_algorithms.co_clustering import CoClustering
    train_affinities = train
    validation_affinities = test
    items = list(set([i for u, i, r in train]))
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    reader = Reader(rating_scale=rating_scale)
    trainset = Dataset.load_from_df(train, reader).build_full_trainset()
    testset = Dataset.load_from_df(test, reader).build_full_trainset().build_testset()
    trainset_for_testing = Dataset.load_from_df(pd.DataFrame(train_affinities),
                                                reader).build_full_trainset().build_testset()

    def use_algo(algo, name):
        start = time.time()
        algo.fit(trainset)
        predictions = algo.test(testset)
        end = time.time()
        total_time = end - start
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        ex_ee = extraction_efficiency(algo, train_affinities, validation_affinities, surprise_get_topk, items)

        train_predictions = algo.test(trainset_for_testing)
        train_rmse = accuracy.rmse(train_predictions, verbose=False)
        train_mae = accuracy.mae(train_predictions, verbose=False)
        train_predictions = [p.est for p in train_predictions]
        predictions = [p.est for p in predictions]
        user_rating_count_metrics = metrics_by_num_interactions_user(train_affinities, validation_affinities,
                                                                     train_predictions, predictions,
                                                                     ex_ee["actuals"], ex_ee["predictions"],
                                                                     ex_ee["train_actuals"], ex_ee["train_predictions"],
                                                                     ex_ee["train_actuals_score_dict"],
                                                                     ex_ee["validation_actuals_score_dict"])
        user_rating_count_metrics["algo"] = name
        stats = {"algo": name, "rmse": rmse, "mae": mae,
                 "train_rmse": train_rmse, "train_mae": train_mae, "time": total_time,
                 "user_rating_count_metrics": user_rating_count_metrics}
        stats.update(ex_ee["metrics"])
        return stats

    algo_map = {"svd": SVD(**(algo_params["svd"] if "svd" in algo_params else {})),
                "svdpp": SVDpp(**(algo_params["svdpp"] if "svdpp" in algo_params else {})),
                "baseline": BaselineOnly(bsl_options={'method': 'sgd'}),
                "clustering": CoClustering(
                    **(algo_params["clustering"] if "clustering" in algo_params else dict(n_cltr_u=5, n_cltr_i=10))),
                "normal": NormalPredictor()}
    results = list(map(lambda a: use_algo(algo_map[a], a), algo))
    user_rating_count_metrics = pd.concat([r["user_rating_count_metrics"] for r in results])
    for s in results:
        del s["user_rating_count_metrics"]
    return None, results, user_rating_count_metrics, None, None


def test_hybrid(train_affinities, validation_affinities, users, items, hyperparameters,
                get_data_mappers, rating_scale, algo,
                enable_error_analysis=False, enable_baselines=False):
    from . import SVDppHybrid, HybridGCNRec, GCNRetriever
    embedding_mapper, user_data, item_data = get_data_mappers()
    kwargs = dict(user_data=user_data, item_data=item_data, hyperparameters=copy.deepcopy(hyperparameters))
    if algo == "svdpp_hybrid":
        recsys = SVDppHybrid(embedding_mapper=embedding_mapper,
                             knn_params=hyperparameters["knn_params"],
                             rating_scale=rating_scale,
                             n_content_dims=hyperparameters["n_dims"],
                             n_collaborative_dims=hyperparameters["n_dims"],
                             fast_inference=False, super_fast_inference=False)
    elif algo in ["gcn_hybrid"]:
        recsys = HybridGCNRec(embedding_mapper=embedding_mapper,
                              knn_params=hyperparameters["knn_params"],
                              rating_scale=rating_scale,
                              n_content_dims=hyperparameters["n_content_dims"],
                              n_collaborative_dims=hyperparameters["n_dims"],
                              fast_inference=False, super_fast_inference=False)
    elif algo in ["gcn_retriever"]:
        recsys = GCNRetriever(embedding_mapper=embedding_mapper,
                              knn_params=hyperparameters["knn_params"],
                              rating_scale=rating_scale,
                              n_content_dims=hyperparameters["n_content_dims"],
                              n_collaborative_dims=hyperparameters["n_dims"],
                              fast_inference=False, super_fast_inference=False)

    start = time.time()
    _, _ = recsys.fit(users, items,
                      train_affinities, **kwargs)
    end = time.time()
    total_time = end - start

    default_preds = recsys.predict([(users[0], "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl"),
                                    (users[0], items[0]),
                                    ("21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl",
                                     "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl"),
                                    ("21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl", items[0])])
    print("Default Preds = ", default_preds)
    assert np.sum(np.isnan(default_preds)) == 0

    recsys.fast_inference = False
    recsys.super_fast_inference = False
    res2 = {"algo": algo, "time": total_time}
    predictions, actuals, stats, user_rating_count_metrics = get_prediction_details(recsys, train_affinities,
                                                                                    validation_affinities,
                                                                                    model_get_topk, items)
    res2.update(stats)
    user_rating_count_metrics["algo"] = res2["algo"]
    results = [res2]

    #
    if enable_baselines:
        recsys.fast_inference = False
        recsys.super_fast_inference = True
        res4 = {"algo": "Super-Fast-%s" % algo, "time": total_time}
        predictions, actuals, stats, urcm = get_prediction_details(recsys, train_affinities,
                                                                   validation_affinities,
                                                                   model_get_topk, items)
        res4.update(stats)
        urcm["algo"] = res4["algo"]
        user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))
        results.append(res4)

        #
        recsys.fast_inference = True
        recsys.super_fast_inference = False
        res = {"algo": "Fast-%s" % algo, "time": total_time}
        predictions, actuals, stats, urcm = get_prediction_details(recsys, train_affinities, validation_affinities,
                                                                   model_get_topk, items)
        res.update(stats)
        urcm["algo"] = res["algo"]
        user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))
        results.append(res)

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(train_affinities, validation_affinities, error_df, "Hybrid")
    return recsys, results, user_rating_count_metrics, predictions, actuals


def test_content_only(train_affinities, validation_affinities, users, items, hyperparameters,
                      get_data_mappers, rating_scale, enable_error_analysis=False):
    from . import ContentRecommendation
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
                                                                                    model_get_topk, items)
    res.update(stats)
    user_rating_count_metrics["algo"] = res["algo"]

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(train_affinities, validation_affinities, error_df, "Hybrid")
    results = [res]
    return recsys, results, user_rating_count_metrics, predictions, actuals


def test_once(train_affinities, validation_affinities, users, items, hyperparamters_dict,
              get_data_mappers, rating_scale, algos,
              enable_error_analysis=False, enable_baselines=False):
    results = []
    recs = []
    assert len(algos) > 0
    algos = set(algos)
    assert len(algos - {"surprise", "content_only", "svdpp_hybrid", "gcn_hybrid", "gcn_retriever"}) == 0
    user_rating_count_metrics = pd.DataFrame([],
                                             columns=["algo", "user_rating_count", "rmse", "mae", "train_map", "map",
                                                      "train_rmse", "train_mae"])
    if "surprise" in algos:
        hyperparameters_surprise = hyperparamters_dict["surprise"]
        _, surprise_results, surprise_user_rating_count_metrics, _, _ = test_surprise(train_affinities,
                                                                                      validation_affinities,
                                                                                      algo=hyperparameters_surprise[
                                                                                          "algos"],
                                                                                      algo_params=hyperparameters_surprise,
                                                                                      rating_scale=rating_scale, )
        results.extend(surprise_results)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, surprise_user_rating_count_metrics))

    if "content_only" in algos:
        hyperparameters = hyperparamters_dict["content_only"]
        content_rec, res, content_user_rating_count_metrics, _, _ = test_content_only(train_affinities,
                                                                                      validation_affinities, users,
                                                                                      items, hyperparameters,
                                                                                      get_data_mappers, rating_scale,
                                                                                      enable_error_analysis=enable_error_analysis)
        results.extend(res)
        recs.append(content_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, content_user_rating_count_metrics))

    if "svdpp_hybrid" in algos:
        hyperparameters = hyperparamters_dict["svdpp_hybrid"]
        svd_rec, res, svdpp_user_rating_count_metrics, _, _ = test_hybrid(train_affinities, validation_affinities,
                                                                          users,
                                                                          items, hyperparameters, get_data_mappers,
                                                                          rating_scale,
                                                                          algo="svdpp_hybrid",
                                                                          enable_error_analysis=enable_error_analysis,
                                                                          enable_baselines=enable_baselines)
        results.extend(res)
        recs.append(svd_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, svdpp_user_rating_count_metrics))

    if "gcn_hybrid" in algos:
        hyperparameters = hyperparamters_dict["gcn_hybrid"]
        gcn_rec, res, gcn_user_rating_count_metrics, _, _ = test_hybrid(train_affinities, validation_affinities, users,
                                                                        items, hyperparameters, get_data_mappers,
                                                                        rating_scale,
                                                                        algo="gcn_hybrid",
                                                                        enable_error_analysis=enable_error_analysis,
                                                                        enable_baselines=enable_baselines)
        results.extend(res)
        recs.append(gcn_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, gcn_user_rating_count_metrics))

    if "gcn_retriever" in algos:
        hyperparameters = hyperparamters_dict["gcn_retriever"]
        gcn_rec, res, gcn_user_rating_count_metrics, _, _ = test_hybrid(train_affinities, validation_affinities, users,
                                                                        items, hyperparameters, get_data_mappers,
                                                                        rating_scale,
                                                                        algo="gcn_retriever",
                                                                        enable_error_analysis=enable_error_analysis,
                                                                        enable_baselines=enable_baselines)
        results.extend(res)
        recs.append(gcn_rec)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, gcn_user_rating_count_metrics))

    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    return recs, results, user_rating_count_metrics


def metrics_by_num_interactions_user(train_affinities: List[Tuple[str, str, float]],
                                     validation_affinities: List[Tuple[str, str, float]],
                                     train_predictions: np.ndarray, val_predictions: np.ndarray,
                                     val_user_topk_actuals: Dict[str, List[str]],
                                     val_user_topk_predictions: Dict[str, List[str]],
                                     train_user_topk_actuals: Dict[str, List[str]],
                                     train_user_topk_predictions: Dict[str, List[str]],
                                     train_actuals_score_dict: Dict[str, Dict[str, float]],
                                     validation_actuals_score_dict: Dict[str, Dict[str, float]],
                                     mode="le",
                                     increments=5):
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
        train_map = np.mean(
            [average_precision(train_user_topk_actuals[u], train_user_topk_predictions[u]) for u in users])
        mean_ap = np.mean([average_precision(val_user_topk_actuals[u], val_user_topk_predictions[u]) for u in users])

        train_mrr = np.mean(
            [reciprocal_rank(train_user_topk_actuals[u], train_user_topk_predictions[u]) for u in users])
        train_ndcg = np.mean([ndcg(train_actuals_score_dict[u], train_user_topk_predictions[u]) for u in users])

        mrr = np.mean([reciprocal_rank(val_user_topk_actuals[u], val_user_topk_predictions[u]) for u in users])
        val_ndcg = np.mean([ndcg(validation_actuals_score_dict[u], val_user_topk_predictions[u]) for u in users])

        results.append(["", uc, val_rmse, val_mae, train_map, mean_ap, mrr, val_ndcg, train_rmse, train_mae, train_mrr,
                        train_ndcg])
    results = pd.DataFrame(results,
                           columns=["algo", "user_rating_count", "rmse", "mae", "train_map", "map", "mrr", "ndcg",
                                    "train_rmse", "train_mae", "train_mrr", "train_ndcg"])
    return results


def display_results(results: List[Dict[str, Any]]):
    df = pd.DataFrame.from_records(results)
    df = df.groupby(['algo']).mean()
    df['time'] = df['time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    time = df['retrieval_time']
    df['retrieval_time'] = df['retrieval_time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    from tabulate import tabulate
    from more_itertools import chunked
    col_lists = list(chunked(df.columns, 8))
    for c in col_lists:
        print(tabulate(df[c], headers='keys', tablefmt='psql'))
    df['retrieval_time'] = time
    return df


def visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities):
    # TODO: support which algos have to be plotted?
    validation_users_count = len(set([u for u, i, r in validation_affinities]))
    results['retrieval_time'] = results['retrieval_time'] / validation_users_count
    results['retrieval_time'] = results['retrieval_time'] * 1000
    plt.figure(figsize=(12, 8))
    plt.title("Retrieval Time vs Algorithm")
    sns.barplot(results.index, results.retrieval_time)
    plt.xlabel("Algorithm")
    plt.ylabel("Retrieval Time in milli-seconds")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('retrieval_time_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Mean Absolute Error vs Algorithm")
    sns.barplot(results.index, results.mae)
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('mae_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("RMSE vs Algorithm")
    sns.barplot(results.index, results.rmse)
    plt.xlabel("Algorithm")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('rmse_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("NDCG vs Algorithm")
    sns.barplot(results.index, results["ndcg@100"])
    plt.xlabel("Algorithm")
    plt.ylabel("NDCG")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('ndcg_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Mean Average Precision vs Algorithm")
    sns.barplot(results.index, results.map)
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Average Precision")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('map_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Mean Reciprocal Rank vs Algorithm")
    sns.barplot(results.index, results.mrr)
    plt.xlabel("Algorithm")
    plt.ylabel("Mean Reciprocal Rank ")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('mrr_vs_algo.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Diversity vs Algorithm")
    sns.barplot(results.index, results.diversity)
    plt.xlabel("Algorithm")
    plt.ylabel("Proportion of total items.")
    plt.xticks(rotation=45, ha='right')
    plt.savefig('diversity_vs_algo.png', bbox_inches='tight')

    unique_algos = user_rating_count_metrics["algo"].nunique()
    markers = None
    style = None
    if unique_algos < 6:
        style = 'algo'
        markers = True
    plt.figure(figsize=(12, 8))
    plt.title("Mean Absolute Error vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="mae", hue="algo", markers=markers, style=style,
                 data=user_rating_count_metrics)
    plt.semilogx(basex=2)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('mae_vs_urc.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Mean Average Precision vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="map", hue="algo", markers=markers, style=style,
                 data=user_rating_count_metrics)
    plt.semilogx(basex=2)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('map_vs_urc.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("RMSE vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="rmse", hue="algo", markers=markers, style=style,
                 data=user_rating_count_metrics)
    plt.semilogx(basex=2)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('rmse_vs_urc.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("Mean Reciprocal Rank vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="mrr", hue="algo", markers=markers, style=style,
                 data=user_rating_count_metrics)
    plt.semilogx(basex=2)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('mrr_vs_urc.png', bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    plt.title("NDCG vs User Rating Count")
    sns.lineplot(x="user_rating_count", y="ndcg", hue="algo", markers=markers, style=style,
                 data=user_rating_count_metrics)
    plt.semilogx(basex=2)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('ndcg_vs_urc.png', bbox_inches='tight')


def get_prediction_details(recsys, train_affinities, validation_affinities, model_get_topk, items):
    def get_details(recsys, affinities):
        predictions = np.array(recsys.predict([(u, i) for u, i, r in affinities]))
        if np.sum(np.isnan(predictions)) > 0:
            count = np.sum(np.isnan(predictions))
            raise AssertionError("Encountered Nan Predictions = %s" % count,
                                 np.array(affinities)[np.isnan(predictions)])

        if np.sum(predictions <= 0) > 0:
            indices = predictions <= 0
            count = np.sum(indices)
            raise AssertionError("Encountered Less than 0 Predictions = %s" % count,
                                 list(zip(predictions[indices], np.array(affinities)[indices])))
        actuals = np.array([r for u, i, r in affinities])
        rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
        mae = np.mean(np.abs(actuals - predictions))
        return predictions, actuals, rmse, mae

    predictions, actuals, rmse, mae = get_details(recsys, validation_affinities)
    train_predictions, _, train_rmse, train_mae = get_details(recsys, train_affinities)
    ex_ee = extraction_efficiency(recsys, train_affinities, validation_affinities, model_get_topk, items)
    user_rating_count_metrics = metrics_by_num_interactions_user(train_affinities, validation_affinities,
                                                                 train_predictions, predictions,
                                                                 ex_ee["actuals"], ex_ee["predictions"],
                                                                 ex_ee["train_actuals"], ex_ee["train_predictions"],
                                                                 ex_ee["train_actuals_score_dict"],
                                                                 ex_ee["validation_actuals_score_dict"])
    stats = {"rmse": rmse, "mae": mae,
             "train_rmse": train_rmse, "train_mae": train_mae}
    stats.update(ex_ee["metrics"])
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


def get_small_subset(df_user, df_item, ratings,
                     cores):
    import networkx as nx
    users = list(set([u for u, i, r in ratings.values]))
    items = list(set([i for u, i, r in ratings.values]))
    user_id_to_index = bidict(zip(users, list(range(len(users)))))
    item_id_to_index = bidict(zip(items, list(range(len(users), len(users) + len(items)))))
    G = nx.Graph([(user_id_to_index[u], item_id_to_index[i]) for u, i, r in ratings.values])
    k_core_edges = list(nx.k_core(G, k=cores).edges())
    users = set([user_id_to_index.inverse[u] for u, i in k_core_edges if u in user_id_to_index.inverse])
    items = set([item_id_to_index.inverse[i] for u, i in k_core_edges if i in item_id_to_index.inverse])
    df_user = df_user[df_user.user.isin(set(users))]
    ratings = ratings[(ratings.user.isin(users)) & (ratings.item.isin(items))]
    df_item = df_item[df_item["item"].isin(set(ratings.item))]
    return df_user, df_item, ratings


def run_model_for_hpo(df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale,
                      hyperparameters, algo, report,
                      enable_kfold=False, provided_test_set=True):
    if enable_kfold:
        provided_test_set = False
    rmse, ndcg = run_models_for_testing(df_user, df_item, user_item_affinities,
                                        prepare_data_mappers, rating_scale,
                                        [algo], {algo: hyperparameters}, provided_test_set=provided_test_set,
                                        enable_kfold=enable_kfold, display=False, report=report)

    return rmse, ndcg


def run_models_for_testing(df_user, df_item, user_item_affinities,
                           prepare_data_mappers, rating_scale,
                           algos, hyperparamters_dict,
                           enable_error_analysis=False, enable_baselines=False,
                           enable_kfold=False, display=True, report=lambda x, y: None,
                           provided_test_set=False):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    if not enable_kfold:
        if provided_test_set:
            train_affinities = [(u, i, r) for u, i, r, t in user_item_affinities if not t]
            validation_affinities = [(u, i, r) for u, i, r, t in user_item_affinities if t]
        else:
            train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2,
                                                                       stratify=[u for u, i, r in user_item_affinities])

        recs, results, user_rating_count_metrics = test_once(train_affinities, validation_affinities,
                                                             list(df_user.user.values),
                                                             list(df_item.item.values),
                                                             hyperparamters_dict,
                                                             prepare_data_mappers, rating_scale, algos,
                                                             enable_error_analysis=enable_error_analysis,
                                                             enable_baselines=enable_baselines)
        rmse, ndcg = results[0]['rmse'], results[0]['ndcg@100']

    else:
        X = np.array(user_item_affinities)
        y = np.array([u for u, i, r in user_item_affinities])
        skf = StratifiedKFold(n_splits=5)
        results = []
        user_rating_count_metrics = pd.DataFrame([],
                                                 columns=["algo", "user_rating_count", "rmse", "mae", "map",
                                                          "train_rmse",
                                                          "train_mae"])
        step = 0
        for train_index, test_index in skf.split(X, y):
            train_affinities, validation_affinities = X[train_index], X[test_index]
            train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
            validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
            #
            recs, res, ucrms = test_once(train_affinities, validation_affinities, list(df_user.user.values),
                                         list(df_item.item.values),
                                         hyperparamters_dict,
                                         prepare_data_mappers, rating_scale, algos,
                                         enable_error_analysis=False, enable_baselines=enable_baselines)

            rmse, ndcg = res[0]['rmse'], res[0]['ndcg@100']
            report({"rmse": rmse, "ndcg": ndcg}, step)
            step += 1
            user_rating_count_metrics = pd.concat((user_rating_count_metrics, ucrms))
            results.extend(res)

    if display:
        user_rating_count_metrics = user_rating_count_metrics.groupby(
            ["algo", "user_rating_count"]).mean().reset_index()
        user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
        results = display_results(results)
        results.to_csv("overall_results.csv")
        visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)

    else:
        results = pd.DataFrame.from_records(results)
        results = results.groupby(["algo"]).mean().reset_index()
        rmse, ndcg = results["rmse"].values[0], results["ndcg@100"].values[0]
    return rmse, ndcg
