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

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding, \
    normalize_affinity_scores_by_user
from hwer import Feature, FeatureSet, FeatureType
from hwer import HybridRecommenderSVDpp, SVDppDNN

# tf.compat.v1.disable_eager_execution()

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

# print(ratings[ratings["user_id"]=="1051"])

check_working = True  # Setting to False uses all the data
enable_kfold = False
enable_error_analysis = False
kfold_multiplier = 2 if enable_kfold else 1
verbose = 2 if os.environ.get("LOGLEVEL") in ["DEBUG"] else 0
test_retrieval = False

hyperparameters = dict(combining_factor=0.1,
                       collaborative_params=dict(
                           prediction_network_params=dict(lr=0.05, epochs=10 * kfold_multiplier, batch_size=64,
                                                          network_width=64, padding_length=50,
                                                          network_depth=4 * kfold_multiplier, verbose=verbose,
                                                          kernel_l2=0.0, rating_regularizer=0.0,
                                                          bias_regularizer=0.02, dropout=0.1),
                           item_item_params=dict(lr=0.001, epochs=5 * kfold_multiplier, batch_size=512,
                                                 verbose=verbose),
                           user_user_params=dict(lr=0.001, epochs=5 * kfold_multiplier, batch_size=512,
                                                 verbose=verbose),
                           user_item_params=dict(lr=0.05, epochs=10 * kfold_multiplier, batch_size=128,
                                                 verbose=verbose, margin=0.5)))

if check_working:
    movie_counts = ratings.groupby(["movie_id"])[["user_id"]].count().reset_index()
    movie_counts = movie_counts.sort_values(by="user_id", ascending=False).head(2000)
    movies = movies[movies["movie_id"].isin(movie_counts.movie_id)]

    ratings = ratings[(ratings.movie_id.isin(movie_counts.movie_id))]

    user_counts = ratings.groupby(["user_id"])[["movie_id"]].count().reset_index()
    user_counts = user_counts.sort_values(by="movie_id", ascending=False).head(500)
    ratings = ratings.merge(user_counts[["user_id"]], on="user_id")
    users = users[users["user_id"].isin(user_counts.user_id)]
    ratings = ratings[(ratings.movie_id.isin(movies.movie_id)) & (ratings.user_id.isin(users.user_id))]
    print("Total Samples Present = %s" % (ratings.shape[0]))
    samples = min(100000, ratings.shape[0])
    ratings = ratings.sample(samples)

print("Total Samples Taken = %s" % (ratings.shape[0]))

user_item_affinities = [(row[0], row[1], row[2]) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]
item_list = list(set([i for u, i, r in user_item_affinities]))


def test_once(train_affinities, validation_affinities, items, capabilities=["svdpp", "resnet", "content", "triplet", "implicit"]):
    embedding_mapper = {}
    embedding_mapper['gender'] = CategoricalEmbedding(n_dims=2)
    embedding_mapper['age'] = CategoricalEmbedding(n_dims=2)
    embedding_mapper['occupation'] = CategoricalEmbedding(n_dims=4*kfold_multiplier)
    embedding_mapper['zip'] = CategoricalEmbedding(n_dims=2*kfold_multiplier)

    embedding_mapper['text'] = FlairGlove100AndBytePairEmbedding()
    embedding_mapper['numeric'] = NumericEmbedding(4*kfold_multiplier)
    embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=4*kfold_multiplier)

    u1 = Feature(feature_name="gender", feature_type=FeatureType.CATEGORICAL, values=users.gender.values)
    u2 = Feature(feature_name="age", feature_type=FeatureType.CATEGORICAL, values=users.age.astype(str).values)
    u3 = Feature(feature_name="occupation", feature_type=FeatureType.CATEGORICAL,
                 values=users.occupation.astype(str).values)
    u4 = Feature(feature_name="zip", feature_type=FeatureType.CATEGORICAL, values=users.zip.astype(str).values)
    user_data = FeatureSet([u1, u2, u3, u4])

    i1 = Feature(feature_name="text", feature_type=FeatureType.STR, values=movies.text.values)
    i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=movies.genres.values)
    i3 = Feature(feature_name="numeric", feature_type=FeatureType.NUMERIC,
                 values=movies[["title_length", "overview_length", "runtime"]].values)
    item_data = FeatureSet([i1, i2, i3])

    kwargs = {}
    kwargs['user_data'] = user_data
    kwargs['item_data'] = item_data
    kwargs["hyperparameters"] = copy.deepcopy(hyperparameters)
    kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_svd"] = False
    kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = False
    kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["resnet_content_each_layer"] = False
    kwargs["hyperparameters"]['collaborative_params'][
        "use_triplet"] = False
    kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
        "use_implicit"] = False
    if "svdpp" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_svd"] = True
    if "resnet" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = True
    if "content" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "resnet_content_each_layer"] = True
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "use_content"] = True
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = True
    if "triplet" in capabilities:
        kwargs["hyperparameters"]['collaborative_params'][
            "use_triplet"] = True
    if "implicit" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "use_implicit"] = True
    if "dnn" in capabilities or "resnet" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "use_dnn"] = True
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["lr"] = 0.1

    recsys = SVDppDNN(embedding_mapper=embedding_mapper,
                                    knn_params=dict(n_neighbors=200, index_time_params={'M': 15, 'ef_construction': 200, }),
                                    rating_scale=(1, 5),
                                    n_content_dims=32 * kfold_multiplier,
                                    n_collaborative_dims=32 * kfold_multiplier)

    start = time.time()
    user_vectors, item_vectors = recsys.fit(users.user_id.values, movies.movie_id.values,
                                            train_affinities, **kwargs)
    # cos_sims = []
    # for i in range(len(item_vectors)):
    #     cos_sims.append([])
    #     for j in range(len(item_vectors)):
    #         sim = cos_sim(item_vectors[i], item_vectors[j])
    #         cos_sims[i].append(sim)
    # cos_sims = np.array(cos_sims)
    # print(cos_sims.min(), cos_sims.max(), cos_sims.mean())

    end = time.time()
    total_time = end - start

    res = {"algo":"hybrid-" + "_".join(capabilities), "time": total_time}
    predictions, actuals, stats = get_prediction_details(recsys, train_affinities, validation_affinities, model_get_topk, items)
    res.update(stats)

    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(error_df, "Hybrid")
    results = [res]
    return recsys, results, predictions, actuals


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.25, stratify=users_for_each_rating)
    results = []

    capabilities = []
    recsys, res, predictions, actuals = test_once(train_affinities, validation_affinities, item_list,
                                                  capabilities=capabilities)
    results.extend(res)
    display_results(results)

    capabilities = ["triplet"]
    recsys, res, predictions, actuals = test_once(train_affinities, validation_affinities, item_list,
                                                  capabilities=capabilities)
    results.extend(res)
    display_results(results)

    capabilities = ["triplet", "content"]
    recsys, res, predictions, actuals = test_once(train_affinities, validation_affinities, item_list, capabilities=capabilities)
    results.extend(res)
    display_results(results)

    results.extend(test_surprise(train_affinities, validation_affinities, item_list, algo=["baseline", "svdpp"]))
    display_results(results)
    print(list(zip(actuals[:50], predictions[:50])))
    if test_retrieval:
        user_id = users.user_id.values[0]
        recommendations = recsys.find_items_for_user(user=user_id, positive=[], negative=[])
        res, dist = zip(*recommendations)
        print(recommendations[:10])
        recommended_movies = res[:10]
        recommended_movies = movies[movies['movie_id'].isin(recommended_movies)][
            ["title", "genres", "year", "keywords", "overview"]]
        actual_movies = ratings[ratings.user_id == user_id]["movie_id"].values
        actual_movies = movies[movies['movie_id'].isin(actual_movies)][["title", "genres", "year", "keywords", "overview"]]
        print(recommended_movies)
        print(actual_movies)
else:
    X = np.array(user_item_affinities)
    y = np.array(users_for_each_rating)
    skf = StratifiedKFold(n_splits=5)
    results = []
    for train_index, test_index in skf.split(X, y):
        train_affinities, validation_affinities = X[train_index], X[test_index]
        train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
        validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
        capabilities = []
        recsys, res, _, _ = test_once(train_affinities, validation_affinities, item_list,
                                                      capabilities=capabilities)
        results.extend(res)
        capabilities = ["triplet"]
        recsys, res, _, _ = test_once(train_affinities, validation_affinities, item_list,
                                      capabilities=capabilities)
        results.extend(res)
        capabilities = ["triplet", "content"]
        recsys, res, _, _ = test_once(train_affinities, validation_affinities, item_list,
                                      capabilities=capabilities)
        results.extend(res)
        results.extend(test_surprise(train_affinities, validation_affinities, item_list, algo=["baseline", "svdpp"]))
        display_results(results)

        print("#" * 80)
    display_results(results)
