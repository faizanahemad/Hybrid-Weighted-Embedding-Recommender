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

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding
from hwer import Feature, FeatureSet, FeatureType
from hwer import SVDppHybrid, ContentRecommendation, HybridGCNRec
from hwer import FasttextEmbedding


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
test_data_subset = True
enable_kfold = False
enable_error_analysis = False
verbose = 2 if os.environ.get("LOGLEVEL") in ["DEBUG"] else 0
test_retrieval = False

if test_data_subset:
    cores = 10
    df_user, df_item, ratings = get_small_subset(df_user, df_item, ratings,
                                                 cores)

ratings = ratings[["user", "item", "rating"]]
user_item_affinities = [(row[0], row[1], float(row[2])) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]
user_list = list(df_user.user.values)
item_list = list(df_item.item.values)

min_rating = np.min([r for u, i, r in user_item_affinities])
max_rating = np.max([r for u, i, r in user_item_affinities])
rating_scale = (min_rating, max_rating)

print("Total Samples Taken = %s, |Users| = %s |Items| = %s" % (ratings.shape[0], len(user_list), len(item_list)))

hyperparameter_content = dict(n_dims=40, combining_factor=0.1,
                              knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }))

hyperparameters_svdpp = dict(n_dims=40, combining_factor=0.1,
                             knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                             collaborative_params=dict(
                                 prediction_network_params=dict(lr=0.02, epochs=50, batch_size=128,
                                                                network_width=96, padding_length=50,
                                                                network_depth=4, verbose=verbose,
                                                                kernel_l2=0.002,
                                                                bias_regularizer=0.01, dropout=0.05,
                                                                use_resnet=True, use_content=True),
                                 user_item_params=dict(lr=0.5, epochs=40, batch_size=64,
                                                       verbose=verbose, margin=1.0)))

hyperparameters_gcn = dict(n_dims=40, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.005, epochs=20, batch_size=1024,
                                                              network_width=128, padding_length=50,
                                                              network_depth=3, verbose=verbose,
                                                              kernel_l2=0.002,
                                                              bias_regularizer=0.01, dropout=0.0, use_content=False),
                               user_item_params=dict(lr=0.1, epochs=2, batch_size=64,
                                                     gcn_lr=0.01, gcn_epochs=2, gcn_layers=2, gcn_dropout=0.0,
                                                     gcn_hidden_dims=96,
                                                     gcn_batch_size=int(
                                                         2 ** np.floor(np.log2(len(user_item_affinities) / 20))),
                                                     verbose=verbose, margin=1.0)))

hyperparameters_surprise = {"svdpp": {"n_factors": 10, "n_epochs": 20}, "algos": ["svdpp"]}

hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn, content_only=hyperparameter_content,
                           svdpp_hybrid=hyperparameters_svdpp, surprise=hyperparameters_surprise, )

svdpp_hybrid = False
gcn_hybrid = True
surprise = False
content_only = False


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2, stratify=[u for u, i, r in user_item_affinities])
    recs, results, user_rating_count_metrics = test_once(train_affinities, validation_affinities, user_list, item_list,
                                                         hyperparamters_dict,
                                                         prepare_data_mappers, rating_scale,
                                                         svdpp_hybrid=svdpp_hybrid, gcn_hybrid=gcn_hybrid,
                                                         surprise=surprise, content_only=content_only,
                                                         enable_error_analysis=enable_error_analysis)
    results = display_results(results)
    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    print(user_rating_count_metrics)
    user_rating_count_metrics.to_csv("algo_user_rating_count_%s.csv" % cores, index=False)
    results.reset_index().to_csv("overall_results_%s.csv" % cores, index=False)
    visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
    if test_retrieval:
        recsys = recs[-1]
        user_id = df_user.user.values[0]
        recommendations = recsys.find_items_for_user(user=user_id, positive=[], negative=[])
        res, dist = zip(*recommendations)
        print(recommendations[:10])
        recommended_items = res[:10]
        recommended_items = df_item[df_item['item'].isin(recommended_items)][
            ["gl", "category_code", "sentence", "price"]]
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
    user_rating_count_metrics = pd.DataFrame([],
                                             columns=["algo", "user_rating_count", "rmse", "mae", "map", "train_rmse",
                                                      "train_mae"])
    for train_index, test_index in skf.split(X, y):
        train_affinities, validation_affinities = X[train_index], X[test_index]
        train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
        validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
        #
        recs, res, ucrms = test_once(train_affinities, validation_affinities, user_list, item_list,
                                     hyperparamters_dict, prepare_data_mappers, rating_scale,
                                     svdpp_hybrid=True,
                                     gcn_hybrid=True, surprise=True, content_only=True,
                                     enable_error_analysis=False)

        user_rating_count_metrics = pd.concat((user_rating_count_metrics, ucrms))
        res = display_results(res)
        results.append(res)
        print("#" * 80)

    results = pd.concat(results)
    results = results.groupby(["algo"]).mean().reset_index()
    user_rating_count_metrics = user_rating_count_metrics.groupby(["algo", "user_rating_count"]).mean().reset_index()
    display_results(results)
    visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
