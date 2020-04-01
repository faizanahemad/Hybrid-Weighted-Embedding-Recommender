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

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding
from hwer import Feature, FeatureSet, FeatureType
from hwer import SVDppHybrid
from hwer import FasttextEmbedding
from ast import literal_eval
import numpy as np


def get_data_mapper(df_user, df_item, dataset="100K"):

    def prepare_data_mappers_100K():
        embedding_mapper = {}
        embedding_mapper['categorical'] = CategoricalEmbedding(n_dims=32)
        embedding_mapper['categorical_year'] = CategoricalEmbedding(n_dims=4)

        embedding_mapper['text'] = FlairGlove100AndBytePairEmbedding()
        embedding_mapper['numeric'] = NumericEmbedding(16)
        embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=32)

        u1 = Feature(feature_name="categorical", feature_type=FeatureType.CATEGORICAL,
                     values=df_user[["gender", "age", "occupation", "zip"]].values)
        user_data = FeatureSet([u1])
        df_item.year = "_" + df_item.year.apply(str) + "_"

        i1 = Feature(feature_name="text", feature_type=FeatureType.STR, values=df_item.text.values)
        i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=df_item.genres.values)
        i3 = Feature(feature_name="numeric", feature_type=FeatureType.NUMERIC,
                     values=np.abs(df_item[["title_length", "overview_length", "runtime"]].values) + 1)
        i4 = Feature(feature_name="categorical_year", feature_type=FeatureType.CATEGORICAL,
                     values=df_item.year.values)
        item_data = FeatureSet([i1, i2, i3, i4])
        return embedding_mapper, user_data, item_data

    def prepare_data_mappers_1M():
        embedding_mapper = {}
        embedding_mapper['categorical'] = CategoricalEmbedding(n_dims=32)

        embedding_mapper['text'] = FlairGlove100AndBytePairEmbedding()
        embedding_mapper['numeric'] = NumericEmbedding(16)
        embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=32)

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

    if dataset == "100K":
        return prepare_data_mappers_100K
    elif dataset == "1M":
        return prepare_data_mappers_1M
    elif dataset == "20M":
        pass
    else:
        raise ValueError("Unsupported Dataset")


def get_data_reader(dataset="100K"):

    def read_data_100K(**kwargs):
        users = pd.read_csv("100K/users.csv", sep="\t")
        movies = pd.read_csv("100K/movies.csv", sep="\t")
        # Based on GC-MC Paper and code: https://github.com/riannevdberg/gc-mc/blob/master/gcmc/preprocessing.py#L326
        test_method = kwargs["test_method"] if "test_method" in kwargs else "random-split"
        if test_method == "random-split" or test_method == "stratified-split":
            if "fold" in kwargs and type(kwargs["fold"]) == int and 1 <= kwargs["fold"] <= 5:
                train_file = "100K/ml-100k/u%s.base" % kwargs["fold"]
                test_file = "100K/ml-100k/u%s.test" % kwargs["fold"]
            else:
                train_file = "100K/ml-100k/u1.base"
                test_file = "100K/ml-100k/u1.test"
            train = pd.read_csv(train_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])[["user", "item", "rating"]]
            test = pd.read_csv(test_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])[["user", "item", "rating"]]
        elif test_method == "ncf":
            train_file = "100K/ml-100k/u.data"
            train = pd.read_csv(train_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])
            train["rating"] = 1
            test = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).tail(1))
            train = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).head(-1))
        else:
            raise ValueError()
        train = train[["user", "item", "rating"]]
        test = test[["user", "item", "rating"]]
        train["is_test"] = False
        test["is_test"] = True
        ratings = pd.concat((train, test))
        movies.genres = movies.genres.fillna("[]").apply(literal_eval)
        movies['year'] = movies['year'].fillna(-1).astype(int)
        movies.keywords = movies.keywords.fillna("[]").apply(literal_eval)
        movies.keywords = movies.keywords.apply(lambda x: " ".join(x))
        movies.tagline = movies.tagline.fillna("")
        text_columns = ["title", "keywords", "overview", "tagline", "original_title"]
        movies[text_columns] = movies[text_columns].fillna("")
        movies['text'] = movies["title"] + " " + movies["keywords"] + " " + movies["overview"] + " " + movies[
            "tagline"] + " " + \
                         movies["original_title"]
        movies["title_length"] = movies["title"].apply(len)
        movies["overview_length"] = movies["overview"].apply(len)
        movies["runtime"] = movies["runtime"].fillna(0.0)
        users.rename(columns={"id": "user"}, inplace=True)
        movies.rename(columns={"id": "item"}, inplace=True)
        return users, movies, ratings

    def read_data_1M(**kwargs):
        test_method = kwargs["test_method"] if "test_method" in kwargs else "random-split"
        users = pd.read_csv("1M/users.csv", sep="\t", engine="python")
        movies = pd.read_csv("1M/movies.csv", sep="\t", engine="python")
        ratings = pd.read_csv("1M/ratings.csv", sep="\t", engine="python")

        ratings['movie_id'] = ratings['movie_id'].astype(str)
        ratings['user_id'] = ratings['user_id'].astype(str)
        ratings.rename(columns={"user_id": "user", "movie_id": "item"}, inplace=True)
        # Based on Paper https://arxiv.org/pdf/1605.09477.pdf (CF-NADE GC-MC)
        if test_method == "random-split":
            train = ratings.sample(frac=0.9)
            test = ratings[~ratings.index.isin(train.index)]
        elif test_method == "stratified-split":
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(ratings, test_size=0.1, stratify=ratings["user"])
        elif test_method == "ncf":
            train = ratings
            train["rating"] = 1
            test = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).tail(1))
            train = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).head(-1))
        train = train[["user", "item", "rating"]]
        test = test[["user", "item", "rating"]]
        train["is_test"] = False
        test["is_test"] = True
        ratings = pd.concat((train, test))
        users['user_id'] = users['user_id'].astype(str)
        movies['movie_id'] = movies['movie_id'].astype(str)

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
        movies.rename(columns={"movie_id": "item"}, inplace=True)
        users.rename(columns={"user_id": "user"}, inplace=True)
        return users, movies, ratings

    if dataset == "100K":
        return read_data_100K
    elif dataset == "1M":
        return read_data_1M
    elif dataset == "20M":
        pass
    elif dataset == "netflix":
        pass
    elif dataset == "pinterest":
        pass
    elif dataset == "msd":
        pass
    elif dataset == "douban":
        pass
    else:
        raise ValueError("Unsupported Dataset")


def build_dataset(dataset, **kwargs):
    read_data = get_data_reader(dataset=dataset)
    df_user, df_item, ratings = read_data(**kwargs)

    prepare_data_mappers = get_data_mapper(df_user, df_item, dataset=dataset)
    ratings = ratings.sample(frac=1.0)
    provided_test_set = False
    assert len(ratings.columns) in [3, 4]
    if len(ratings.columns) == 3:
        user_item_affinities = [(row[0], row[1], float(row[2])) for row in ratings.values]
    if len(ratings.columns) == 4:
        provided_test_set = True
        user_item_affinities = [(row[0], row[1], float(row[2]), bool(row[3])) for row in ratings.values]

    rating_scale = (np.min([x[2] for x in user_item_affinities]), np.max([x[2] for x in user_item_affinities]))
    return df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale, provided_test_set