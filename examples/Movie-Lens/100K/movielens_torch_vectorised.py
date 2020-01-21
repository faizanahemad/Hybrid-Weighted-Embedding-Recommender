import os
import pandas as pd
import numpy as np
import re
import stanfordnlp
import torch
import dgl
import string
import tqdm
from functools import partial
from ast import literal_eval


def ml100k_enhanced_reader(directory):
    users = pd.read_csv("users.csv", sep="\t").set_index('id')
    movies = pd.read_csv("movies.csv", sep="\t").set_index('id')
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

    ratings = pd.read_csv("ratings.csv", sep="\t")
    return users, movies, ratings


class MovieLens(object):

    def __init__(self, dataset, directory, feature_size=200, split_by_time=None):
        """

        :param directory:
        :param split_by_time: Set this to 'timestamp' to perform a user-based temporal split of the dataset.
        """
        self.split_by_time = split_by_time
        self.feature_size = feature_size
        if dataset == 'ml-100k':
            users, products, ratings = ml100k_enhanced_reader(directory)
        elif dataset == 'ml-100k_enhanced':
            pass

        self.products = products
        self.users = users
        self.ratings = ratings
        self.ratings = self.data_split(self.ratings)
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.products = self.products[self.products.index.isin(self.ratings['product_id'])]

        self.build_graph()
        self.generate_mask()
        self.generate_candidates()

    # Change this field to 'timestamp' to perform a user-based temporal split of the dataset,
    # where the ratings of each users are ordered by timestamps.  The first 80% of the ordered
    # data falls into the training set, and the middle 10% and last 10% goes to validation and
    # test set respectively.
    # If this field is None, the ratings of each users would be shuffled and splitted in a
    # ratio of 80-10-10.
    def split_user(self, df, filter_counts=0, timestamp=None):
        df_new = df.copy()
        df_new['prob'] = -1

        df_new_sub = (df_new['product_count'] >= filter_counts).to_numpy().nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        if timestamp is not None and timestamp in df_new.columns:
            df_new = df_new.iloc[df_new_sub].sort_values(timestamp)
            df_new['prob'] = prob
            return df_new
        else:
            np.random.shuffle(prob)
            df_new['prob'].iloc[df_new_sub] = prob
            return df_new

    def data_split(self, ratings):
        ratings = ratings.groupby('user_id', group_keys=False).apply(
            partial(self.split_user, filter_counts=10, timestamp=self.split_by_time))
        ratings['train'] = ratings['prob'] <= 0.8
        ratings['valid'] = (ratings['prob'] > 0.8) & (ratings['prob'] <= 0.9)
        ratings['test'] = ratings['prob'] > 0.9
        ratings.drop(['prob'], axis=1, inplace=True)
        return ratings

    # process the features and build the DGL graph
    def build_graph(self):
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(len(user_ids) + len(product_ids))

        #
        self.users['zip'] = self.users['zip'].apply(lambda x: str(x)[:-3])

        from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, \
            NumericEmbedding
        from hwer import Feature, FeatureSet, FeatureType
        embedding_mapper = {}
        embedding_mapper['categorical'] = CategoricalEmbedding(n_dims=16)
        embedding_mapper['categorical_year'] = CategoricalEmbedding(n_dims=2)

        embedding_mapper['text'] = FlairGlove100AndBytePairEmbedding()
        embedding_mapper['numeric'] = NumericEmbedding(8)
        embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=8)

        u1 = Feature(feature_name="categorical", feature_type=FeatureType.CATEGORICAL,
                     values=self.users[["gender", "age", "occupation", "zip"]].values)
        user_data = FeatureSet([u1])
        self.products.year = "_" + self.products.year.apply(str) + "_"

        i1 = Feature(feature_name="text", feature_type=FeatureType.STR, values=self.products.text.values)
        i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=self.products.genres.values)
        i3 = Feature(feature_name="numeric", feature_type=FeatureType.NUMERIC,
                     values=np.abs(self.products[["title_length", "overview_length", "runtime"]].values) + 1)
        i4 = Feature(feature_name="categorical_year", feature_type=FeatureType.CATEGORICAL, values=self.products.year.values)
        item_data = FeatureSet([i1, i2, i3, i4])

        from hwer import ContentRecommendation
        kwargs = dict(user_data=user_data, item_data=item_data, hyperparameters=dict(n_dims=self.feature_size, combining_factor=0.1,))
        recsys = ContentRecommendation(embedding_mapper=embedding_mapper,
                                       knn_params=dict(n_neighbors=5, index_time_params={'M': 15, 'ef_construction': 5, }),
                                       rating_scale=(1, 5), n_output_dims=self.feature_size, )
        train_affinities = self.ratings[self.ratings['train']][["user_id", "product_id", "rating"]].values
        user_vectors, product_vectors = recsys.fit(self.users.index, self.products.index,
                                                   train_affinities, **kwargs)

        g.ndata['content'] = torch.FloatTensor(np.concatenate((user_vectors, product_vectors)))

        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                                   for id_ in self.ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices

        g.add_edges(
            rating_user_vertices,
            rating_product_vertices,
            data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8),
                  'rating': torch.FloatTensor(self.ratings['rating'])})
        g.add_edges(
            rating_product_vertices,
            rating_user_vertices,
            data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8),
                  'rating': torch.FloatTensor(self.ratings['rating'])})
        self.g = g

    # Assign masks of training, validation and test set onto the DGL graph
    # according to the rating table.
    def generate_mask(self):
        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(self.ratings['train'].values.astype('uint8'))
        edge_data = {
            'valid': valid_tensor,
            'test': test_tensor,
            'train': train_tensor,
        }

        self.g.edges[self.rating_user_vertices, self.rating_product_vertices].data.update(edge_data)
        self.g.edges[self.rating_product_vertices, self.rating_user_vertices].data.update(edge_data)

    # Generate the list of products for each user in training/validation/test set.
    def generate_candidates(self):
        self.p_train = []
        self.p_valid = []
        self.p_test = []
        for uid in tqdm.tqdm(self.user_ids):
            user_ratings = self.ratings[self.ratings['user_id'] == uid]
            self.p_train.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['train']]['product_id'].values]))
            self.p_valid.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['valid']]['product_id'].values]))
            self.p_test.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['test']]['product_id'].values]))