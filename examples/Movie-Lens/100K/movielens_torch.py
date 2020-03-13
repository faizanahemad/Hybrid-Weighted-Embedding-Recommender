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


def reduce_zip(zip):
    try:
        zip = int(int(zip)/100)
    except ValueError:
        zip = 0
    return zip


def reduce_year(year):
    try:
        year = int(int(year)/5)
    except ValueError:
        year = 0
    return year


def ml100k_default_reader(directory):
    def read_user_line(l):
        id_, age, gender, occupation, zip_ = l.strip().split('|')
        age = np.searchsorted([20, 30, 40, 50, 60], age)  # bin the ages into <20, 20-30, 30-40, ..., >60
        return {'id': int(id_), 'gender': gender, 'age': age, 'occupation': occupation, 'zip': reduce_zip(zip_)}

    def read_product_line(l):
        fields = l.strip().split('|')
        id_ = fields[0]
        title = fields[1]
        genres = fields[-19:]

        # extract year
        if re.match(r'.*\([0-9]{4}\)$', title):
            year = reduce_year(title[-5:-1])
            title = title[:-6].strip()
        else:
            year = 0

        data = {'id': int(id_), 'title': title, 'year': year}
        for i, g in enumerate(genres):
            data['genre' + str(i)] = (int(g) != 0)
        return data

    def read_rating_line(l):
        user_id, product_id, rating, timestamp = l.split()
        return {'user_id': int(user_id), 'product_id': int(product_id), 'rating': float(rating),
                'timestamp': int(timestamp)}

    users = []
    products = []
    ratings = []

    # read ratings
    with open(os.path.join(directory, 'u1.base')) as f:
        for l in f:
            rating = read_rating_line(l)
            rating["test"] = False
            rating["train"] = True
            ratings.append(rating)
    with open(os.path.join(directory, 'u1.test')) as f:
        for l in f:
            rating = read_rating_line(l)
            rating["test"] = True
            rating["train"] = False
            ratings.append(rating)

    ratings = pd.DataFrame(ratings)
    product_count = ratings['product_id'].value_counts()
    product_count.name = 'product_count'
    ratings = ratings.join(product_count, on='product_id')

    # read users - if user feature does not exist, we find all unique user IDs
    # appeared in the rating table and create an empty table from that.
    user_file = os.path.join(directory, 'u.user')
    with open(user_file) as f:
        for l in f:
            users.append(read_user_line(l))
    users = pd.DataFrame(users).set_index('id').astype('category')

    # read products
    with open(os.path.join(directory, 'u.item'), encoding='latin1') as f:
        for l in f:
            products.append(read_product_line(l))
    products = (
        pd.DataFrame(products)
            .set_index('id')
            .astype({'year': 'category'}))
    movie_extra_features = pd.read_csv("movies.csv", engine="python", sep="\t")
    movie_extra_features = movie_extra_features[["id", "overview", "keywords", "tagline", "runtime"]]
    from ast import literal_eval
    movie_extra_features["keywords"] = movie_extra_features["keywords"].fillna("[]").apply(literal_eval).apply(
        lambda x: " ".join(x))
    movie_extra_features[["overview", "tagline"]] = movie_extra_features[["overview", "tagline"]].fillna(" ")
    movie_extra_features["runtime"] = movie_extra_features["runtime"].fillna(movie_extra_features["runtime"].mean())
    products = products.merge(movie_extra_features, right_on="id", left_index=True)

    genres = products.columns[products.dtypes == bool]
    products["title"] = products["title"] + " " + products["overview"] + " " + products["tagline"] + " " + products["keywords"]
    products.drop(columns=["overview", "tagline", "keywords", "id"], inplace=True)
    return users, products, ratings, genres


class MovieLens(object):

    def __init__(self, directory, split_by_time=None):
        """

        :param directory:
        :param split_by_time: Set this to 'timestamp' to perform a user-based temporal split of the dataset.
        """
        self.split_by_time = split_by_time
        users, products, ratings, genres = ml100k_default_reader(directory)
        self.genres = genres
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

        # user features
        for user_column in self.users.columns:
            udata = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            udata[:len(user_ids)] = \
                    torch.LongTensor(self.users[user_column].cat.codes.values.astype('int64') + 1)
            g.ndata[user_column] = udata

        # product genre
        product_genres = torch.from_numpy(self.products[self.genres].values.astype('float32'))
        g.ndata['genre'] = torch.zeros(g.number_of_nodes(), len(self.genres))
        g.ndata['genre'][len(user_ids):len(user_ids) + len(product_ids)] = product_genres

        # product year
        if 'year' in self.products.columns:
            g.ndata['year'] = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            g.ndata['year'][len(user_ids):len(user_ids) + len(product_ids)] = \
                    torch.LongTensor(self.products['year'].cat.codes.values.astype('int64') + 1)

        # product title
        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []
        for t in tqdm.tqdm(self.products['title'].values):
            doc = nlp(t)
            words = set()
            for s in doc.sentences:
                words.update(w.lemma.lower() for w in s.words
                             if not re.fullmatch(r'['+string.punctuation+']+', w.lemma))
            vocab.update(words)
            title_words.append(words)
        vocab = list(vocab)
        vocab_invmap = {w: i for i, w in enumerate(vocab)}
        # bag-of-words
        g.ndata['title'] = torch.zeros(g.number_of_nodes(), len(vocab))
        for i, tw in enumerate(tqdm.tqdm(title_words)):
            g.ndata['title'][i, [vocab_invmap[w] for w in tw]] = 1
        self.vocab = vocab
        self.vocab_invmap = vocab_invmap
        
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
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(self.ratings['train'].values.astype('uint8'))
        edge_data = {
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.rating_user_vertices, self.rating_product_vertices].data.update(edge_data)
        self.g.edges[self.rating_product_vertices, self.rating_user_vertices].data.update(edge_data)
        
    # Generate the list of products for each user in training/validation/test set.
    def generate_candidates(self):
        self.p_train = []
        self.p_test = []
        for uid in tqdm.tqdm(self.user_ids):
            user_ratings = self.ratings[self.ratings['user_id'] == uid]
            self.p_train.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['train']]['product_id'].values]))
            self.p_test.append(np.array(
                [self.product_ids_invmap[i] for i in user_ratings[user_ratings['test']]['product_id'].values]))