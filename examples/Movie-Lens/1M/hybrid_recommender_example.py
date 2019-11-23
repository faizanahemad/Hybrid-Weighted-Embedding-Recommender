from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Tuple, Sequence, Type, Set, Optional, Any
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, SVDpp
from surprise import accuracy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from ast import literal_eval

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding, \
    FlairGlove100Embedding
from hwer import Feature, FeatureSet, FeatureType
from hwer import HybridRecommender, HybridRecommenderSVDpp
from hwer.utils import cos_sim

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

check_working = False  # Setting to False uses all the data
enable_kfold = True
if check_working:
    movie_counts = ratings.groupby(["movie_id"])[["user_id"]].count().reset_index()
    movie_counts = movie_counts.sort_values(by="user_id", ascending=False).head(200)
    movies = movies[movies["movie_id"].isin(movie_counts.movie_id)]

    ratings = ratings[(ratings.movie_id.isin(movie_counts.movie_id))]

    user_counts = ratings.groupby(["user_id"])[["movie_id"]].count().reset_index()
    user_counts = user_counts.sort_values(by="movie_id", ascending=False).head(100)
    ratings = ratings.merge(user_counts[["user_id"]], on="user_id")
    users = users[users["user_id"].isin(user_counts.user_id)]
    ratings = ratings[(ratings.movie_id.isin(movies.movie_id)) & (ratings.user_id.isin(users.user_id))]
    samples = min(5000, ratings.shape[0])
    ratings = ratings.sample(samples)
    print("Total Samples Taken = %s" % (ratings.shape[0]))

user_item_affinities = [(row[0], row[1], row[2]) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]


def test_surprise(train, test, algo=["baseline","svdpp"], algo_params={}, rating_scale=(1, 5)):
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
                "train_rmse": train_rmse, "train_mae": train_mae, "time":total_time}

    algo_map = {"svd": SVD(**(algo_params["svd"] if "svd" in algo_params else {})),
                "svdpp": SVDpp(**(algo_params["svdpp"] if "svdpp" in algo_params else {})),
                "baseline": BaselineOnly(bsl_options={'method':'sgd'})}
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


def test_once(train_affinities, validation_affinities):
    embedding_mapper = {}
    embedding_mapper['gender'] = CategoricalEmbedding(n_dims=1)
    embedding_mapper['age'] = CategoricalEmbedding(n_dims=1)
    embedding_mapper['occupation'] = CategoricalEmbedding(n_dims=2)
    embedding_mapper['zip'] = CategoricalEmbedding(n_dims=2)

    embedding_mapper['text'] = FlairGlove100Embedding()
    embedding_mapper['numeric'] = NumericEmbedding(2)
    embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=2)

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

    kfold_multiplier = 1 if enable_kfold else 1

    kwargs = {}
    kwargs['user_data'] = user_data
    kwargs['item_data'] = item_data
    kwargs["hyperparameters"] = dict(combining_factor=0.5,
                                     collaborative_params=dict(
                                         prediction_network_params=dict(lr=0.001, epochs=10*kfold_multiplier, batch_size=512,
                                                                        network_width=2,
                                                                        network_depth=2*kfold_multiplier, verbose=0,
                                                                        kernel_l1=0.0, kernel_l2=0.01,
                                                                        activity_l1=0.0, activity_l2=0.001,
                                                                        bias_regularizer=0.1, dropout=0.2),
                                         item_item_params=dict(lr=0.001, epochs=5*kfold_multiplier, batch_size=512, network_width=4,
                                                               network_depth=2*kfold_multiplier, verbose=0, kernel_l1=0.0,
                                                               kernel_l2=0.01,
                                                               activity_l1=0.0, activity_l2=0.0, dropout=0.2),
                                         user_user_params=dict(lr=0.001, epochs=5*kfold_multiplier, batch_size=512, network_width=4,
                                                               network_depth=2*kfold_multiplier, verbose=0, kernel_l1=0.0,
                                                               kernel_l2=0.01,
                                                               activity_l1=0.0, activity_l2=0.0, dropout=0.2),
                                         user_item_params=dict(lr=0.001, epochs=5*kfold_multiplier, batch_size=512, network_width=2,
                                                               network_depth=2*kfold_multiplier, verbose=0, kernel_l1=0.0,
                                                               kernel_l2=0.01,
                                                               activity_l1=0.0, activity_l2=0.0, dropout=0.2)))

    results = test_surprise(train_affinities, validation_affinities)
    recsys = HybridRecommenderSVDpp(embedding_mapper=embedding_mapper, knn_params=None, rating_scale=(1, 5),
                               n_content_dims=8*kfold_multiplier,
                               n_collaborative_dims=8*kfold_multiplier)
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
    _, actuals, train_rmse, train_mae = get_prediction_details(recsys, train_affinities)
    results.append({"algo": "hybrid", "rmse": rmse, "mae": mae,
                    "train_rmse": train_rmse, "train_mae": train_mae, "time": total_time})
    display_results(results)
    return recsys, results, predictions, actuals


if not enable_kfold:
    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.25)
    recsys, results, predictions, actuals = test_once(train_affinities, validation_affinities)
    print(list(zip(actuals[:20], predictions[:20])))
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
        recsys, res, _, _ = test_once(train_affinities, validation_affinities)
        results.extend(res)
        print("#"*80)
    display_results(results)



