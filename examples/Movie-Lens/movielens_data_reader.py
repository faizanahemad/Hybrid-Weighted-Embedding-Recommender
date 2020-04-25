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

from hwer import CategoricalEmbed, FlairGlove100AndBytePairEmbed, NumericEmbed, Node, Edge
from hwer.utils import merge_dicts_nested, build_row_dicts
from ast import literal_eval
import numpy as np


def process_age(age):
    age = np.searchsorted([20, 30, 40, 50, 60], age)
    return age


def process_zip(zip):
    try:
        zip = int(zip)
        zip = int(zip / 10)
    except ValueError:
        zip = -1
    return zip


def process_zip_vectorized(zip):
    return np.vectorize(process_zip)(zip)


def get_data_mapper(df_user, df_item, dataset="100K"):

    def prepare_data_mappers_100K():
        user_nodes = [Node("user", n) for n in df_user.user.values]
        n_users = len(user_nodes)
        df_user['age_processed'] = process_age(df_user['age'])
        df_user['zip_1'] = process_zip_vectorized(df_user['zip'])
        df_user['zip_2'] = process_zip_vectorized(df_user['zip_1'])
        user_data = dict(zip(user_nodes, build_row_dicts("categorical", df_user[["gender", "age_processed", "occupation", "zip_1", "zip_2"]].values)))
        user_numeric = dict(zip(user_nodes, build_row_dicts("numeric", df_user[["user_rating_mean", "user_rating_count"]].values)))
        user_data = merge_dicts_nested(user_data, user_numeric)

        item_nodes = [Node("item", i) for i in df_item.item.values]
        n_items = len(item_nodes)
        df_item.year_processed = "_" + df_item.year.apply(str) + "_"
        item_text = dict(zip(item_nodes, build_row_dicts("text", df_item.text.values)))
        item_cats = dict(zip(item_nodes, build_row_dicts("categorical", df_item[["year_processed", "genres"]].values)))
        item_numerics = dict(zip(item_nodes, build_row_dicts("numeric", np.abs(df_item[["title_length", "overview_length", "runtime", "item_rating_mean", "item_rating_count"]].values))))

        item_data = merge_dicts_nested(item_text, item_cats, item_numerics)
        assert len(user_data) == n_users
        assert len(item_data) == n_items

        node_data = dict(user_data)
        node_data.update(item_data)
        embedding_mapper = dict(user=dict(categorical=CategoricalEmbed(n_dims=32), numeric=NumericEmbed(32)),
                                item=dict(text=FlairGlove100AndBytePairEmbed(), categorical=CategoricalEmbed(n_dims=32), numeric=NumericEmbed(32)))
        return embedding_mapper, node_data

    if dataset == "100K":
        return prepare_data_mappers_100K
    elif dataset == "1M":
        return prepare_data_mappers_100K
    elif dataset == "20M":
        pass
    else:
        raise ValueError("Unsupported Dataset")


def get_data_reader(dataset="100K"):

    def process_100K_1M(users, movies, train, test):
        train = train[["user", "item", "rating", "timestamp"]]
        test = test[["user", "item", "rating", "timestamp"]]
        user_stats = train.groupby(["user"])["rating"].agg(["mean", "count"]).reset_index()
        item_stats = train.groupby(["item"])["rating"].agg(["mean", "count"]).reset_index()
        user_stats.rename(columns={"mean": "user_rating_mean", "count": "user_rating_count"}, inplace=True)
        item_stats.rename(columns={"mean": "item_rating_mean", "count": "item_rating_count"}, inplace=True)

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

        users = users.merge(user_stats, how="left", on="user")
        movies = movies.merge(item_stats, how="left", on="item")
        movies = movies.fillna(movies.mean())
        users = users.fillna(users.mean())
        return users, movies, ratings

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
            train = pd.read_csv(train_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])
            test = pd.read_csv(test_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])
        elif test_method == "ncf":
            train_file = "100K/ml-100k/u.data"
            train = pd.read_csv(train_file, sep="\t", header=None, names=["user", "item", "rating", "timestamp"])
            train["rating"] = 1
            test = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).tail(1))
            train = train.groupby('user', group_keys=False).apply(lambda x: x.sort_values(["timestamp"]).head(-1))
        else:
            raise ValueError()
        return process_100K_1M(users, movies, train, test)

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
        return process_100K_1M(users, movies, train, test)

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


def get_graph_builder(dataset):

    def build_graph_100K(df_user, df_item, ratings):
        ratings = ratings.sample(frac=1.0)
        user_nodes = [Node("user", n) for n in df_user.user.values]
        item_nodes = [Node("item", i) for i in df_item.item.values]
        nodes = user_nodes + item_nodes
        assert len(ratings.columns) == 5
        ts = ratings.groupby(["user"])[["timestamp"]].agg(["min", "max"]).reset_index()
        ts.columns = ["user", "min", "max"]
        ts["range"] = ts["max"] - ts["min"]
        ratings = ratings.merge(ts[["user", "min", "range"]], on="user")
        ratings["timestamp"] = (ratings["timestamp"] - ratings["min"])/ratings["range"]
        ratings["rating"] = ratings["timestamp"] + ratings["rating"]
        ratings = ratings[["user", "item", "rating", "is_test"]]
        user_item_affinities = [(Node("user", row[0]), Node("item", row[1]),
                                 float(row[2]), bool(row[3])) for row in ratings.values]
        edges = [(Edge(src, dst, weight), train_test) for src, dst, weight, train_test in
                                user_item_affinities]

        df_user['age_processed'] = process_age(df_user['age'])
        df_user['zip_1'] = process_zip_vectorized(df_user['zip'])
        df_user['zip_2'] = process_zip_vectorized(df_user['zip_1'])
        df_item["year_processed"] = "_" + df_item.year.apply(str) + "_"

        weight = 0.25
        user_age_edges = [(Node("user", u), Node("age", age), weight, False) for u, age in df_user[['user', 'age_processed']].values]
        age_nodes = list(set([age for _, age, _, _ in user_age_edges]))
        user_age_edges = [(Edge(s, d, w), is_test) for s, d, w, is_test in user_age_edges]
        nodes = nodes + age_nodes
        edges = edges + user_age_edges
        #
        user_zip_1_edges = [(Node("user", u), Node("zip_1", zip_1), weight, False) for u, zip_1 in df_user[['user', 'zip_1']].values]
        zip_1_nodes = list(set([zip_1 for _, zip_1, _, _ in user_zip_1_edges]))
        user_zip_1_edges = [(Edge(s, d, w), is_test) for s, d, w, is_test in user_zip_1_edges]
        nodes = nodes + zip_1_nodes
        edges = edges + user_zip_1_edges
        #
        user_zip_2_edges = [(Node("user", u), Node("zip_2", zip_2), weight, False) for u, zip_2 in
                            df_user[['user', 'zip_2']].values]
        zip_2_nodes = list(set([zip_2 for _, zip_2, _, _ in user_zip_2_edges]))
        user_zip_2_edges = [(Edge(s, d, w), is_test) for s, d, w, is_test in user_zip_2_edges]
        nodes = nodes + zip_2_nodes
        edges = edges + user_zip_2_edges
        #
        item_year_edges = [(Node("item", i), Node("year", year), weight, False) for i, year in
                            df_item[['item', 'year_processed']].values]
        year_nodes = list(set([year for _, year, _, _ in item_year_edges]))
        item_year_edges = [(Edge(s, d, w), is_test) for s, d, w, is_test in item_year_edges]
        nodes = nodes + year_nodes
        edges = edges + item_year_edges

        item_genre_edges = [(Node("item", i), Node("genre", genre), weight, False) for i, genres in
                           df_item[['item', 'genres']].values for genre in genres]
        genre_nodes = list(set([genre for _, genre, _, _ in item_genre_edges]))
        item_genre_edges = [(Edge(s, d, w), is_test) for s, d, w, is_test in item_genre_edges]
        nodes = nodes + genre_nodes
        edges = edges + item_genre_edges
        node_types = {'user', 'item', 'age', 'zip_1', 'zip_2', 'year', 'genre'}

        return nodes, edges, node_types

    if dataset == "100K":
        return build_graph_100K
    elif dataset == "1M":
        return build_graph_100K
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
    graph_builder = get_graph_builder(dataset)
    nodes, edges, node_types = graph_builder(df_user, df_item, ratings)
    return nodes, edges, node_types, prepare_data_mappers