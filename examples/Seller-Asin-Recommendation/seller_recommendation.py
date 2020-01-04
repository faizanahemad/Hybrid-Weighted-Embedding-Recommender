import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from hwer.validation import *
from hwer.utils import average_precision, cos_sim
import networkx as nx

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
from sklearn.utils import shuffle

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding, \
    normalize_affinity_scores_by_user
from hwer import Feature, FeatureSet, FeatureType
from hwer import SVDppHybrid, ContentRecommendation, HybridGCNRec
from hwer import FasttextEmbedding


def read_data():
    df_item = pd.read_csv("small_items.csv")
    df_item["item"] = df_item["item"].astype(str)

    df_user = pd.read_csv("small_users.csv")
    df_user["user"] = df_user["user"].astype(str)

    user_item_affinity = pd.read_csv("small_user_item_affinity.csv")
    user_item_affinity["affinity"] = user_item_affinity["affinity"].astype(float)
    user_item_affinity.rename(columns={"affinity": "rating"}, inplace=True)
    user_item_affinity["user"] = user_item_affinity["user"].astype(str)
    user_item_affinity["item"] = user_item_affinity["item"].astype(str)
    ratings = user_item_affinity

    df_user[["min_item_price", "avg_item_price", "num_asins"]] = np.clip(
        df_user[["min_item_price", "avg_item_price", "num_asins"]], 1, 1e6)

    df_item["browse_node_l2"] = df_item["browse_node_l2"].apply(lambda x: str(x).split())
    df_user["browse_node_l2"] = df_user["browse_node_l2"].apply(lambda x: str(x).split())

    df_item["subcategory_code"] = df_item["subcategory_code"].apply(lambda x: str(x).split())
    df_user["subcategory_code"] = df_user["subcategory_code"].apply(lambda x: str(x).split())
    return df_user, df_item, ratings


df_user, df_item, ratings = read_data()
#
check_working = True
enable_kfold = False
enable_error_analysis = False
verbose = 2 if os.environ.get("LOGLEVEL") in ["DEBUG"] else 0
test_retrieval = False
cores = 10
# Diff init, low lr, high lr
hyperparameters = dict(combining_factor=0.2,
                       collaborative_params=dict(
                           prediction_network_params=dict(lr=0.03, epochs=10, batch_size=128,
                                                          network_width=96, padding_length=50,
                                                          network_depth=4, verbose=verbose,
                                                          kernel_l2=0.001,
                                                          bias_regularizer=0.01, dropout=0.05),
                           user_item_params=dict(lr=0.5, epochs=20, batch_size=64,
                                                 gcn_lr=0.005, gcn_epochs=20, gcn_layers=3,
                                                 verbose=verbose, margin=1.0)))

if check_working:
    G = nx.Graph([(u, i) for u, i, r in ratings.values if r >= 0])
    k_core_edges = list(nx.k_core(G, k=cores).edges())
    users = set([u for u, i in k_core_edges])
    items = set([i for u, i in k_core_edges])
    df_user = df_user[df_user.user.isin(set(users))]
    negatives = ratings[(ratings.user.isin(users))]
    ratings = ratings[(ratings.user.isin(users)) & (ratings.item.isin(items))]
    negatives = negatives[negatives.rating < 0]
    if len(negatives) > 0:
        negatives = negatives.sort_values(["item"])
        negatives = negatives.groupby('user', group_keys=False).apply(lambda x: x.head(min(5, len(x))))
        ratings = pd.concat((negatives, ratings)).sample(frac=1.0)
    df_item = df_item[df_item["item"].isin(set(ratings.item))]
    print("Total Samples Present = %s" % (ratings.shape[0]))

ratings = ratings[["user", "item", "rating"]]
user_item_affinities = [(row[0], row[1], float(row[2])) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]
item_list = list(set([i for u, i, r in user_item_affinities if r > 1e-4]))
user_list = list(set([u for u, i, r in user_item_affinities]))


min_rating = np.min([r for u, i, r in user_item_affinities])
max_rating = np.max([r for u, i, r in user_item_affinities])
rating_scale = (min_rating, max_rating)

print("Total Samples Taken = %s, |Users| = %s |Items| = %s" % (ratings.shape[0], len(user_list), len(item_list)))


def prepare_data_mappers():
    i1 = Feature(feature_name="item_numeric", feature_type=FeatureType.NUMERIC,
                 values=df_item[["price", "customer_average_review_rating", "customer_review_count"]].fillna(0).values)
    i2 = Feature(feature_name="sentence", feature_type=FeatureType.STR, values=df_item["sentence"].fillna("").values)
    i3 = Feature(feature_name="categorical", feature_type=FeatureType.CATEGORICAL,
                 values=df_item[["gl", "browse_root_name", "category_code"]].fillna("__NULL__").values)
    i4 = Feature(feature_name="browse_node_l2", feature_type=FeatureType.MULTI_CATEGORICAL,
                 values=df_item.browse_node_l2.values)
    i5 = Feature(feature_name="subcategory_code", feature_type=FeatureType.MULTI_CATEGORICAL,
                 values=df_item.subcategory_code.values)
    item_data = FeatureSet([i1, i2, i3, i4, i5])

    u1 = Feature(feature_name="user_numeric", feature_type=FeatureType.NUMERIC,
                 values=df_user[["min_item_price", "avg_item_price", "num_asins"]].fillna(0).values)
    u2 = Feature(feature_name="sentence", feature_type=FeatureType.STR, values=df_user["sentence"].fillna("").values)
    u3 = Feature(feature_name="browse_node_l2", feature_type=FeatureType.MULTI_CATEGORICAL,
                 values=df_user.browse_node_l2.values)
    u4 = Feature(feature_name="subcategory_code", feature_type=FeatureType.MULTI_CATEGORICAL,
                 values=df_user.subcategory_code.values)

    user_data = FeatureSet([u1, u2, u3, u4])

    embedding_mapper = {}

    embedding_mapper['item_numeric'] = NumericEmbedding(2, log=False, n_iters=20)
    embedding_mapper['user_numeric'] = NumericEmbedding(2, log=False, n_iters=20)
    embedding_mapper['categorical'] = CategoricalEmbedding(n_dims=4, n_iters=20)
    embedding_mapper['browse_node_l2'] = MultiCategoricalEmbedding(n_dims=4, n_iters=20)
    embedding_mapper['subcategory_code'] = MultiCategoricalEmbedding(n_dims=4, n_iters=20)
    embedding_mapper['sentence'] = FasttextEmbedding(n_dims=24, fasttext_file="fasttext_new.bin")

    return embedding_mapper, user_data, item_data


def test_once(train_affinities, validation_affinities, items, capabilities=["resnet", "content"]):
    embedding_mapper, user_data, item_data = prepare_data_mappers()
    kwargs = {}
    kwargs['user_data'] = user_data
    kwargs['item_data'] = item_data
    kwargs["hyperparameters"] = copy.deepcopy(hyperparameters)
    kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = False
    if "resnet" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = True
    if "content" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "use_content"] = True
    recsys = HybridGCNRec(embedding_mapper=embedding_mapper,
                         knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                         rating_scale=rating_scale,
                         n_content_dims=40,
                         n_collaborative_dims=40,
                         fast_inference=False, content_only_inference=False)

    start = time.time()
    _, _ = recsys.fit(df_user.user.values, df_item.item.values,
                                            train_affinities, **kwargs)
    end = time.time()
    total_time = end - start

    embedding_mapper, user_data, item_data = prepare_data_mappers()
    kwargs['user_data'] = user_data
    kwargs['item_data'] = item_data
    content_recsys = ContentRecommendation(embedding_mapper=embedding_mapper,
                                           knn_params=dict(n_neighbors=200,
                                                           index_time_params={'M': 15, 'ef_construction': 200, }),
                                           rating_scale=rating_scale, n_output_dims=32)
    _, _ = content_recsys.fit(df_user.user.values, df_item.item.values,
                              train_affinities, **kwargs)

    assert np.sum(np.isnan(recsys.predict([(user_list[0], "21120eifjcchchbninlkkgjnjjegrjbldkidbuunfjghbdhfl")]))) == 0

    recsys.fast_inference = True
    recsys.content_only_inference = False
    res2 = {"algo": "Fast-SVDPP", "time": total_time}
    predictions, actuals, stats, user_rating_count_metrics = get_prediction_details(recsys, train_affinities, validation_affinities,
                                                         model_get_topk, items, min_positive_rating=0.0, ignore_below_rating=0.0)
    res2.update(stats)
    user_rating_count_metrics["algo"] = res2["algo"]

    res4 = {"algo": "Content-Only", "time": total_time}
    _, _, stats, urcm = get_prediction_details(content_recsys, train_affinities, validation_affinities,
                                         model_get_topk, items, min_positive_rating=0.0, ignore_below_rating=0.0)
    res4.update(stats)
    urcm["algo"] = res4["algo"]
    user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))

    recsys.fast_inference = False
    recsys.content_only_inference = False
    res = {"algo":"Hybrid", "time": total_time}
    predictions, actuals, stats, urcm = get_prediction_details(recsys, train_affinities, validation_affinities, model_get_topk, items, min_positive_rating=0.1, ignore_below_rating=0.0)
    res.update(stats)
    urcm["algo"] = res["algo"]
    user_rating_count_metrics = pd.concat((urcm, user_rating_count_metrics))

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(train_affinities, validation_affinities, error_df, "Hybrid")
    results = [res, res2, res4]
    return recsys, results, user_rating_count_metrics, predictions, actuals


def stratified_split(user_item_affinities):
    positive_samples = [[u, i, float(r)] for u, i, r in user_item_affinities if r >= 0]
    negative_samples = [[u, i, float(r)] for u, i, r in user_item_affinities if r < 0]
    train_pos, val_pos = train_test_split(positive_samples, test_size=0.2, stratify=[u for u, i, r in positive_samples])
    print(len(negative_samples))
    if len(negative_samples) > 0:
        train_neg, val_neg = train_test_split(negative_samples, test_size=0.2, stratify=[u for u, i, r in negative_samples])
        train_affinities = np.concatenate((train_pos, train_neg))
        validation_affinities = np.concatenate((val_pos, val_neg))
    else:
        train_affinities = train_pos
        validation_affinities = val_pos
    print("Total Train Set = %s, Test Set = %s, Positive Test Set = %s" % (len(train_affinities), len(validation_affinities), len(val_pos)))
    train_affinities = shuffle(train_affinities)
    validation_affinities = shuffle(validation_affinities)
    val_pos = shuffle(val_pos)
    train_affinities = [[u, i, float(r)] for u, i, r in train_affinities]
    validation_affinities = [[u, i, float(r)] for u, i, r in validation_affinities]
    val_pos = [[u, i, float(r)] for u, i, r in val_pos]
    return train_affinities, validation_affinities, val_pos


if not enable_kfold:
    train_affinities, validation_affinities, val_pos = stratified_split(user_item_affinities)
    surprise_results = test_surprise(train_affinities, validation_affinities, item_list,
                                     algo=["svdpp"], algo_params={"svdpp": {"n_factors": 10}},
                                     rating_scale=rating_scale, min_positive_rating=0.0, ignore_below_rating=0.0)
    results = []

    capabilities = ["resnet", "content"]
    recsys, res, user_rating_count_metrics, predictions, actuals = test_once(train_affinities, validation_affinities, item_list,
                                                  capabilities=capabilities)
    results.extend(res)
    display_results(results)

    ucrms = [s["user_rating_count_metrics"] for s in surprise_results]
    ucrms = pd.concat(ucrms)
    user_rating_count_metrics = pd.concat((user_rating_count_metrics, ucrms))
    for s in surprise_results:
        del s["user_rating_count_metrics"]
    results.extend(surprise_results)
    results = display_results(results)
    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    print(user_rating_count_metrics)
    user_rating_count_metrics.to_csv("algo_user_rating_count_%s.csv" % cores, index=False)
    results.reset_index().to_csv("overall_results_%s.csv" % cores, index=False)
    visualize_results(results, user_rating_count_metrics, len(user_item_affinities))
    print(list(zip(actuals[:50], predictions[:50])))
    if test_retrieval:
        user_id = df_user.user.values[0]
        recommendations = recsys.find_items_for_user(user=user_id, positive=[], negative=[])
        res, dist = zip(*recommendations)
        print(recommendations[:10])
        recommended_items = res[:10]
        recommended_items = df_item[df_item['item'].isin(recommended_items)][["gl", "category_code", "sentence", "price"]]
        actual_items = ratings[ratings.user == user_id]["item"].sample(10).values
        actual_items = df_item[df_item['item'].isin(actual_items)][["gl", "category_code", "sentence", "price"]]
        print(recommended_items)
        print("=" * 100)
        print(actual_items)
else:
    X = np.array(user_item_affinities)
    y = np.array(users_for_each_rating)
    skf = StratifiedKFold(n_splits=5)
    results = []
    for train_index, test_index in skf.split(X, y):
        train_affinities, validation_affinities = X[train_index], X[test_index]
        train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
        validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
        #
        capabilities = ["content", "resnet"]
        recsys, res, user_rating_count_metrics, predictions, actuals = test_once(train_affinities,
                                                                                 validation_affinities, item_list,
                                                                                 capabilities=capabilities)
        results.extend(res)
        surprise_results = test_surprise(train_affinities, validation_affinities, item_list,
                                         algo=["svdpp"], algo_params={"svdpp": {"n_factors": 10}},
                                         rating_scale=rating_scale, min_positive_rating=0.0, ignore_below_rating=0.0)
        ucrms = [s["user_rating_count_metrics"] for s in surprise_results]
        ucrms = pd.concat(ucrms)
        user_rating_count_metrics = pd.concat((user_rating_count_metrics, ucrms))
        for s in surprise_results:
            del s["user_rating_count_metrics"]
        results.extend(surprise_results)
        user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
        display_results(results)
        print("#" * 80)

    results = results.groupby(["algo"]).mean().reset_index()
    user_rating_count_metrics = user_rating_count_metrics.groupby(["algo", "user_rating_count"]).mean().reset_index()
    display_results(results)
    visualize_results(results, user_rating_count_metrics, len(user_item_affinities))
