import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
import os
import copy

warnings.filterwarnings('ignore')
from typing import List, Dict, Any
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

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding
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

hyperparameters = dict(combining_factor=0.5,
                       collaborative_params=dict(
                           prediction_network_params=dict(lr=0.1, epochs=10 * kfold_multiplier, batch_size=16,
                                                          network_width=128, padding_length=50,
                                                          network_depth=3 * kfold_multiplier, verbose=verbose,
                                                          kernel_l2=0.0, rating_regularizer=0.0,
                                                          bias_regularizer=0.02, dropout=0.0),
                           item_item_params=dict(lr=0.001, epochs=5 * kfold_multiplier, batch_size=512,
                                                 network_depth=2 * kfold_multiplier, verbose=verbose,
                                                 kernel_l2=0.01, dropout=0.0),
                           user_user_params=dict(lr=0.001, epochs=5 * kfold_multiplier, batch_size=512,
                                                 network_depth=2 * kfold_multiplier, verbose=verbose,
                                                 kernel_l2=0.01, dropout=0.0),
                           user_item_params=dict(lr=0.1, epochs=2 * kfold_multiplier, batch_size=256,
                                                 network_depth=2 * kfold_multiplier, verbose=verbose,
                                                 kernel_l2=0.001, dropout=0.0)))

if check_working:
    movie_counts = ratings.groupby(["movie_id"])[["user_id"]].count().reset_index()
    movie_counts = movie_counts.sort_values(by="user_id", ascending=False).head(100)
    movies = movies[movies["movie_id"].isin(movie_counts.movie_id)]

    ratings = ratings[(ratings.movie_id.isin(movie_counts.movie_id))]

    user_counts = ratings.groupby(["user_id"])[["movie_id"]].count().reset_index()
    user_counts = user_counts.sort_values(by="movie_id", ascending=False).head(500)
    ratings = ratings.merge(user_counts[["user_id"]], on="user_id")
    users = users[users["user_id"].isin(user_counts.user_id)]
    ratings = ratings[(ratings.movie_id.isin(movies.movie_id)) & (ratings.user_id.isin(users.user_id))]
    samples = min(10000, ratings.shape[0])
    ratings = ratings.sample(samples)

print("Total Samples Taken = %s" % (ratings.shape[0]))

user_item_affinities = [(row[0], row[1], row[2]) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]


def test_surprise(train, test, algo=["baseline", "svd", "svdpp"], algo_params={}, rating_scale=(1, 5)):
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    reader = Reader(rating_scale=rating_scale)
    trainset = Dataset.load_from_df(train, reader).build_full_trainset()
    # testset = Dataset.load_from_df(test, reader).build_full_trainset().build_anti_testset()
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

        predictions = algo.test(trainset_for_testing)
        train_rmse = accuracy.rmse(predictions, verbose=False)
        train_mae = accuracy.mae(predictions, verbose=False)
        return {"algo": name, "rmse": rmse, "mae": mae,
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
    print(df)


def get_prediction_details(recsys, affinities):
    predictions = recsys.predict([(u, i) for u, i, r in affinities])
    actuals = np.array([r for u, i, r in affinities])
    rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
    mae = np.mean(np.abs(actuals - predictions))
    return predictions, actuals, rmse, mae


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


def test_once(train_affinities, validation_affinities, capabilities=["svdpp", "resnet", "content", "triplet", "implicit"]):
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
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"]["use_resnet"] = True
    if "triplet" in capabilities:
        kwargs["hyperparameters"]['collaborative_params'][
            "use_triplet"] = True
    if "implicit" in capabilities:
        kwargs["hyperparameters"]['collaborative_params']["prediction_network_params"][
            "use_implicit"] = True

    recsys = SVDppDNN(embedding_mapper=embedding_mapper, knn_params=None, rating_scale=(1, 5),
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
    predictions, actuals, rmse, mae = get_prediction_details(recsys, validation_affinities)
    _, _, train_rmse, train_mae = get_prediction_details(recsys, train_affinities)
    if enable_error_analysis:
        error_df = pd.DataFrame({"errors": actuals - predictions, "actuals": actuals, "predictions": predictions})
        error_analysis(error_df, "Hybrid")
    results = [{"algo":"hybrid-" + "_".join(capabilities), "rmse": rmse, "mae": mae,
                "train_rmse": train_rmse, "train_mae": train_mae, "time": total_time}]
    return recsys, results, predictions, actuals


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.25, stratify=users_for_each_rating)

    capabilities = ["resnet", "content", "triplet"]
    recsys, results, predictions, actuals = test_once(train_affinities, validation_affinities, capabilities=capabilities)
    display_results(results)
    #
    # capabilities = ["svdpp", "resnet", "content"]
    # recsys, res, predictions, actuals = test_once(train_affinities, validation_affinities, capabilities=capabilities)
    # results.extend(res)
    # display_results(results)

    # capabilities = []
    # recsys, res, predictions, actuals = test_once(train_affinities, validation_affinities, capabilities=capabilities)
    # results.extend(res)
    # display_results(results)

    results.extend(test_surprise(train_affinities, validation_affinities, algo=["baseline", "svd", "svdpp"]))
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
        recsys, res, _, _ = test_once(train_affinities, validation_affinities, algo="hybrid-resnet")
        # recsys, res1, _, _ = test_once(train_affinities, validation_affinities, algo="hybrid")
        # recsys, res2, _, _ = test_once(train_affinities, validation_affinities, algo="hybrid-triplet")
        # recsys, res3, _, _ = test_once(train_affinities, validation_affinities, algo="hybrid-svdpp")
        # recsys, res4, _, _ = test_once(train_affinities, validation_affinities, algo="hybrid-resnet-content")
        # res.extend(res1)
        # res.extend(res2)
        # res.extend(res3)
        # res.extend(res4)
        res.extend(test_surprise(train_affinities, validation_affinities))
        display_results(res)
        results.extend(res)
        print("#" * 80)
    display_results(results)
