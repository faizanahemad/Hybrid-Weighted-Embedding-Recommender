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

# tf.compat.v1.disable_eager_execution()


def prepare_data_mappers():
    embedding_mapper = {}
    embedding_mapper['categorical'] = CategoricalEmbedding(n_dims=4)

    embedding_mapper['text'] = FlairGlove100AndBytePairEmbedding()
    embedding_mapper['numeric'] = NumericEmbedding(4)
    embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=4)

    u1 = Feature(feature_name="categorical", feature_type=FeatureType.CATEGORICAL, values=df_user[["gender", "age", "occupation", "zip"]].values)
    user_data = FeatureSet([u1])

    # print("Numeric Nans = \n", np.sum(df_item[["title_length", "overview_length", "runtime"]].isna()))
    # print("Numeric Zeros = \n", np.sum(df_item[["title_length", "overview_length", "runtime"]] <= 0))

    i1 = Feature(feature_name="text", feature_type=FeatureType.STR, values=df_item.text.values)
    i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=df_item.genres.values)
    i3 = Feature(feature_name="numeric", feature_type=FeatureType.NUMERIC,
                 values=np.abs(df_item[["title_length", "overview_length", "runtime"]].values) + 1)
    item_data = FeatureSet([i1, i2, i3])
    return embedding_mapper, user_data, item_data


def read_data():
    users = pd.read_csv("users.csv", sep="\t", engine="python")
    movies = pd.read_csv("movies.csv", sep="\t", engine="python")
    ratings = pd.read_csv("ratings.csv", sep="\t", engine="python")

    users['user_id'] = users['user_id'].astype(str)
    movies['movie_id'] = movies['movie_id'].astype(str)
    ratings['movie_id'] = ratings['movie_id'].astype(str)
    ratings['user_id'] = ratings['user_id'].astype(str)

    movies.genres = movies.genres.fillna("[]").apply(literal_eval)
    movies['year'] = movies['year'].fillna(-1).astype(int)
    movies.keywords = movies.keywords.fillna("[]").apply(literal_eval)
    movies.keywords = movies.keywords.apply(lambda x: " ".join(x))
    movies.tagline = movies.tagline.fillna("")
    text_columns = ["title", "keywords", "overview", "tagline", "original_title"]
    movies[text_columns] = movies[text_columns].fillna("")
    movies['text'] = movies["title"] + " " + movies["keywords"] + " " + movies["overview"] + " " + movies["tagline"] + " " + \
                     movies["original_title"]
    movies["title_length"] = movies["title"].apply(len)
    movies["overview_length"] = movies["overview"].apply(len)
    movies["runtime"] = movies["runtime"].fillna(0.0)
    ratings = ratings[["user_id", "movie_id", "rating"]]
    ratings.rename(columns={"user_id": "user", "movie_id": "item"}, inplace=True)
    movies.rename(columns={"movie_id": "item"}, inplace=True)
    users.rename(columns={"user_id": "user"}, inplace=True)
    return users, movies, ratings


df_user, df_item, ratings = read_data()
#
test_data_subset = True
enable_kfold = False
enable_error_analysis = False
verbose = 2 # if os.environ.get("LOGLEVEL") in ["DEBUG", "INFO"] else 0
test_retrieval = False

if test_data_subset:
    cores = 23
    # # ratings = ratings[ratings.rating.isin([1, 5])]
    # item_counts = ratings.groupby(['item'])['user'].count().reset_index()
    # item_counts = item_counts[(item_counts["user"] <= 200) & (item_counts["user"] >= 20)].head(100)
    # items = set(item_counts["item"])
    # ratings = ratings[ratings["item"].isin(items)]
    #
    # user_counts = ratings.groupby(['user'])['item'].count().reset_index()
    # user_counts = user_counts[(user_counts["item"] <= 100) & (user_counts["item"] >= 20)].head(100)
    # users = set(user_counts["user"])
    # ratings = ratings[ratings["user"].isin(users)]
    #
    # # ratings = pd.concat((ratings[ratings.rating == 1].head(2), ratings[ratings.rating == 5].head(3)))
    # users = set(ratings["user"])
    # items = set(ratings["item"])
    # df_user = df_user[df_user["user"].isin(ratings.user)]
    # df_item = df_item[df_item["item"].isin(ratings.item)]
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

hyperparameters_svdpp = dict(n_dims=40, combining_factor=0.1,
                             knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                             collaborative_params=dict(
                                 prediction_network_params=dict(lr=0.75, epochs=20, batch_size=64,
                                                                network_width=96, padding_length=50,
                                                                network_depth=4, verbose=verbose,
                                                                kernel_l2=0.001,
                                                                bias_regularizer=0.002, dropout=0.05,
                                                                use_resnet=True, use_content=True),
                                 user_item_params=dict(lr=0.1, epochs=10, batch_size=64, l2=0.01,
                                                       verbose=verbose, margin=1.0)))

hyperparameters_gcn = dict(n_dims=48, combining_factor=0.1,
                           knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                           collaborative_params=dict(
                               prediction_network_params=dict(lr=0.01, epochs=30, batch_size=1024, padding_length=50,
                                                              network_depth=3, verbose=verbose,
                                                              kernel_l2=1e-8, dropout=0.1, use_content=True, enable_implicit=True),
                               user_item_params=dict(lr=0.2, epochs=5, batch_size=64, l2=0.01,
                                                     gcn_lr=0.01, gcn_epochs=5, gcn_layers=0, gcn_dropout=0.0,
                                                     gcn_kernel_l2=1e-9,
                                                     gcn_batch_size=1024,
                                                     verbose=verbose, margin=0.75)))

hyperparameters_surprise = {"svdpp": {"n_factors": 20, "n_epochs": 20},
                            "svd": {"biased": True, "n_factors": 10},
                            "algos": ["baseline", "svd", "svdpp"]}

hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn, content_only=hyperparameter_content,
                           svdpp_hybrid=hyperparameters_svdpp, surprise=hyperparameters_surprise, )

svdpp_hybrid = False
gcn_hybrid = True
surprise = False
content_only = False


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2, stratify=[u for u, i, r in user_item_affinities])
    print("Train Length = ", len(train_affinities))
    print("Validation Length =", len(validation_affinities))
    recs, results, user_rating_count_metrics = test_once(train_affinities, validation_affinities, user_list, item_list,
                                                         hyperparamters_dict,
                                                         prepare_data_mappers, rating_scale,
                                                         svdpp_hybrid=svdpp_hybrid, gcn_hybrid=gcn_hybrid,
                                                         surprise=surprise, content_only=content_only,
                                                         enable_error_analysis=enable_error_analysis)
    results = display_results(results)
    user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
    # print(user_rating_count_metrics)
    # user_rating_count_metrics.to_csv("algo_user_rating_count_%s.csv" % cores, index=False)
    # results.reset_index().to_csv("overall_results_%s.csv" % cores, index=False)
    # visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
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
