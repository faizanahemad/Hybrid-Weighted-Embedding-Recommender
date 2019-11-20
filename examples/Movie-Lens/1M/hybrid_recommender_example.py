from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding, \
    FlairGlove100Embedding
from hwer import Feature, FeatureSet, FeatureType
from hwer import HybridRecommender

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

check_working = True  # Setting to False does KFold CV for 5 folds
if check_working:
    movies = movies.sample(50)
    users = users.sample(50)
    ratings = ratings[(ratings.movie_id.isin(movies.movie_id))&(ratings.user_id.isin(users.user_id))]
    samples = min(5000, ratings.shape[0])
    ratings = ratings.sample(samples)


user_item_affinities = [(row[0], row[1], row[2]) for row in ratings.values]
users_for_each_rating = [row[0] for row in ratings.values]

if check_working:
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

    kwargs = {}
    kwargs['user_data'] = user_data
    kwargs['item_data'] = item_data
    kwargs["hyperparameters"] = dict(combining_factor=0.5,
                                     collaborative_params=dict(
                                         prediction_network_params=dict(lr=0.001, epochs=1, batch_size=512,
                                                                        network_width=2, network_depth=3),
                                         item_item_params=dict(lr=0.001, epochs=1, batch_size=512, network_width=2,
                                                               network_depth=2),
                                         user_user_params=dict(lr=0.001, epochs=1, batch_size=512, network_width=2,
                                                               network_depth=2),
                                         user_item_params=dict(lr=0.001, epochs=1, batch_size=512, network_width=2,
                                                               network_depth=2)))

    train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.8)
    recsys = HybridRecommender(embedding_mapper=embedding_mapper, knn_params=None, rating_scale=(1, 5),
                               n_content_dims=32,
                               n_collaborative_dims=32)
    user_vectors, item_vectors = recsys.fit(users.user_id.values, movies.movie_id.values,
                                            train_affinities, **kwargs)

    predictions = recsys.predict([(u, i) for u, i, r in validation_affinities[:10]])
    actuals = np.array([r for u, i, r in validation_affinities[:10]])

    print(list(zip(actuals, predictions)))

    print(np.sqrt(np.mean(np.square(actuals - predictions))))

    user_id = users.user_id.values[0]
    recommendations = recsys.find_items_for_user(user=user_id, positive=[], negative=[])
    res, dist = zip(*recommendations)
    print(recommendations[:10])
    recommended_movies = res[:10]
    recommended_movies = movies[movies['movie_id'].isin(recommended_movies)][["title","genres","year","keywords","overview"]]
    actual_movies = ratings[ratings.user_id == user_id]["movie_id"].values
    actual_movies = movies[movies['movie_id'].isin(actual_movies)][["title","genres","year","keywords","overview"]]
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

        kwargs = {}
        kwargs['user_data'] = user_data
        kwargs['item_data'] = item_data

        kwargs["hyperparameters"] = dict(combining_factor=0.5,
                                         collaborative_params=dict(prediction_network_params=dict(lr=0.001, epochs=5, batch_size=512, network_width=2, network_depth=3),
                                                                   item_item_params=dict(lr=0.001, epochs=5, batch_size=512, network_width=2, network_depth=3),
                                                                   user_user_params=dict(lr=0.001, epochs=5, batch_size=512, network_width=2, network_depth=3),
                                                                   user_item_params=dict(lr=0.001, epochs=5, batch_size=512, network_width=2, network_depth=3)))

        recsys = HybridRecommender(embedding_mapper=embedding_mapper, knn_params=None, rating_scale=(1, 5),
                                   n_content_dims=32,
                                   n_collaborative_dims=32)
        _, _ = recsys.fit(users.user_id.values, movies.movie_id.values,
                                                train_affinities, **kwargs)

        predictions = recsys.predict([(u, i) for u, i, r in validation_affinities])
        actuals = np.array([r for u, i, r in validation_affinities])
        rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
        mae = np.mean(np.abs(actuals - predictions))
        print("RMSE = %.4f, MAE = %.4f" % (rmse, mae))
        results.append([rmse, mae])
    results = pd.DataFrame(results, columns=["rmse", "mae"])
    print(results)
    mean_rmse, mean_mae = results.mean().values
    print("Final RMSE = %.4f, MAE = %.4f"%(mean_rmse, mean_mae))
