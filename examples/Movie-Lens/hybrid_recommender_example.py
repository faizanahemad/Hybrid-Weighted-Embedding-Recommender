import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from hwer.validation import *
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
from ast import literal_eval

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding
from hwer import Feature, FeatureSet, FeatureType
from hwer import SVDppHybrid
from hwer import FasttextEmbedding

import movielens_data_reader as mdr


read_data = mdr.get_data_reader(dataset="100K")
df_user, df_item, ratings = read_data()
prepare_data_mappers = mdr.get_data_mapper(df_user, df_item, dataset="100K")

#
test_data_subset = False
enable_kfold = False
enable_error_analysis = False
verbose = 2 # if os.environ.get("LOGLEVEL") in ["DEBUG", "INFO"] else 0

if test_data_subset:
    if False:
        item_counts = ratings.groupby(['item'])['user'].count().reset_index()
        item_counts = item_counts[(item_counts["user"] <= 200) & (item_counts["user"] >= 20)].head(100)
        items = set(item_counts["item"])
        ratings = ratings[ratings["item"].isin(items)]

        user_counts = ratings.groupby(['user'])['item'].count().reset_index()
        user_counts = user_counts[(user_counts["item"] <= 100) & (user_counts["item"] >= 20)].head(100)
        users = set(user_counts["user"])
        ratings = ratings[ratings["user"].isin(users)]

        # ratings = pd.concat((ratings[ratings.rating == 1].head(2), ratings[ratings.rating == 5].head(3)))
        users = set(ratings["user"])
        items = set(ratings["item"])
        df_user = df_user[df_user["user"].isin(ratings.user)]
        df_item = df_item[df_item["item"].isin(ratings.item)]
    else:
        cores = 10
        df_user, df_item, ratings = get_small_subset(df_user, df_item, ratings, cores)

ratings = ratings[["user", "item", "rating"]]
user_item_affinities = [(row[0], row[1], float(row[2])) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]
user_list = list(df_user.user.values)
item_list = list(df_item.item.values)

min_rating = np.min([r for u, i, r in user_item_affinities])
max_rating = np.max([r for u, i, r in user_item_affinities])
rating_scale = (min_rating, max_rating)

print("Total Samples Taken = %s, |Users| = %s |Items| = %s, Rating scale = %s" % (ratings.shape[0], len(user_list), len(item_list), rating_scale))

hyperparameter_content = dict(n_dims=40, combining_factor=0.1,
                              knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }))

hyperparameters_svdpp = dict(n_dims=48, combining_factor=0.1,
                             knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                             collaborative_params=dict(
                                 prediction_network_params=dict(lr=0.5, epochs=35, batch_size=64,
                                                                network_width=128, padding_length=50,
                                                                network_depth=4, verbose=verbose,
                                                                kernel_l2=1e-5,
                                                                bias_regularizer=0.001, dropout=0.05,
                                                                use_resnet=True, use_content=True),
                                 user_item_params=dict(lr=0.1, epochs=10, batch_size=64, l2=0.001,
                                                       verbose=verbose, margin=1.0)))

hyperparameters_gcn = dict(n_dims=48, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.0005, epochs=15, batch_size=1024, padding_length=50,
                                                              network_depth=2, verbose=verbose,
                                                              kernel_l2=1e-6, dropout=0.2, use_content=True, enable_implicit=False),
                               user_item_params=dict(lr=0.2, epochs=5, batch_size=64, l2=0.001,
                                                     gcn_lr=0.001, gcn_epochs=10, gcn_layers=2, gcn_dropout=0.0,
                                                     gcn_kernel_l2=1e-6,
                                                     gcn_batch_size=1024,
                                                     verbose=verbose, margin=0.75)))

hyperparameters_gcn_implicit = dict(n_dims=48, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.001, epochs=15, batch_size=1024, padding_length=50,
                                                              network_depth=2, verbose=verbose,
                                                              kernel_l2=1e-6, dropout=0.2, use_content=True, enable_implicit=True),
                               user_item_params=dict(lr=0.2, epochs=5, batch_size=64, l2=0.001,
                                                     gcn_lr=0.001, gcn_epochs=10, gcn_layers=2, gcn_dropout=0.0,
                                                     gcn_kernel_l2=1e-6,
                                                     gcn_batch_size=1024,
                                                     verbose=verbose, margin=0.75)))


hyperparameters_gcn_deep = dict(n_dims=48, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.001, epochs=15, batch_size=1024, padding_length=50,
                                                              network_depth=2, verbose=verbose,
                                                              kernel_l2=1e-6, dropout=0.2, use_content=True, deep_mode=True),
                               user_item_params=dict(lr=0.2, epochs=5, batch_size=64, l2=0.001,
                                                     gcn_lr=0.001, gcn_epochs=10, gcn_layers=2, gcn_dropout=0.0,
                                                     gcn_kernel_l2=1e-6,
                                                     gcn_batch_size=1024,
                                                     verbose=verbose, margin=0.75)))

hyperparameters_gcn_implicit_deep = dict(n_dims=48, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.001, epochs=15, batch_size=1024, padding_length=50,
                                                              network_depth=2, verbose=verbose,
                                                              kernel_l2=1e-6, dropout=0.2, use_content=True,
                                                              deep_mode=True, enable_implicit=True),
                               user_item_params=dict(lr=0.2, epochs=5, batch_size=64, l2=0.001,
                                                     gcn_lr=0.001, gcn_epochs=10, gcn_layers=2, gcn_dropout=0.0,
                                                     gcn_kernel_l2=1e-6,
                                                     gcn_batch_size=1024,
                                                     verbose=verbose, margin=0.75)))

hyperparameters_surprise = {"svdpp": {"n_factors": 20, "n_epochs": 20},
                            "svd": {"biased": True, "n_factors": 5},
                            "algos": ["baseline", "svd", "svdpp"]}

hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn, gcn_hybrid_implicit=hyperparameters_gcn_implicit,
                           gcn_hybrid_deep=hyperparameters_gcn_deep, gcn_hybrid_implicit_deep=hyperparameters_gcn_implicit_deep,
                           content_only=hyperparameter_content,
                           svdpp_hybrid=hyperparameters_svdpp, surprise=hyperparameters_surprise, )

svdpp_hybrid = False
surprise = True
content_only = False
gcn_hybrid = True
gcn_hybrid_implicit = False
gcn_hybrid_deep = False
gcn_hybrid_implicit_deep = False


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2, stratify=[u for u, i, r in user_item_affinities])
    print("Train Length = ", len(train_affinities))
    print("Validation Length =", len(validation_affinities))
    recs, results, user_rating_count_metrics = test_once(train_affinities, validation_affinities, user_list, item_list,
                                                         hyperparamters_dict,
                                                         prepare_data_mappers, rating_scale,
                                                         svdpp_hybrid=svdpp_hybrid, gcn_hybrid=gcn_hybrid,
                                                         gcn_hybrid_implicit=gcn_hybrid_implicit,
                                                         gcn_hybrid_deep=gcn_hybrid_deep, gcn_hybrid_implicit_deep=gcn_hybrid_implicit_deep,
                                                         surprise=surprise, content_only=content_only,
                                                         enable_error_analysis=enable_error_analysis)
    results = display_results(results)
    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    # print(user_rating_count_metrics)
    # user_rating_count_metrics.to_csv("algo_user_rating_count_%s.csv" % cores, index=False)
    # results.reset_index().to_csv("overall_results_%s.csv" % cores, index=False)
    visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
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
                                     svdpp_hybrid=svdpp_hybrid, gcn_hybrid=gcn_hybrid,
                                     gcn_hybrid_implicit=gcn_hybrid_implicit,
                                     gcn_hybrid_deep=gcn_hybrid_deep, gcn_hybrid_implicit_deep=gcn_hybrid_implicit_deep,
                                     surprise=surprise, content_only=content_only,
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
