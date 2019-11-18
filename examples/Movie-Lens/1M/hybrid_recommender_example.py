from hwer.utils import normalize_affinity_scores_by_user_item, normalize_affinity_scores_by_user

from hwer.utils import unit_length, build_user_item_dict, build_item_user_dict, cos_sim, shuffle_copy
from hwer import HybridRecommender
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Sequence, Type, Set, Optional

from surprise import Dataset
from surprise import accuracy
import pandas as pd
from pathlib import Path

from surprise.model_selection import train_test_split
import numpy as np
from tqdm import tqdm,tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns

users = pd.read_csv("users.csv", sep="\t", engine="python")
movies = pd.read_csv("movies.csv", sep="\t", engine="python")
ratings = pd.read_csv("ratings.csv", sep="\t", engine="python")

users['user_id'] = users['user_id'].astype(str)
movies['movie_id'] = movies['movie_id'].astype(str)
ratings['movie_id'] = ratings['movie_id'].astype(str)
ratings['user_id'] = ratings['user_id'].astype(str)

from ast import literal_eval

movies.genres = movies.genres.fillna("[]").apply(literal_eval)
movies['year'] = movies['year'].fillna(-1).astype(int)

movies.keywords = movies.keywords.fillna("[]").apply(literal_eval)
movies.keywords = movies.keywords.apply(lambda x: " ".join(x))

movies.tagline = movies.tagline.fillna("")
text_columns = ["title","keywords","overview","tagline","original_title"]
movies[text_columns] = movies[text_columns].fillna("")

movies['text'] = movies["title"] +" "+ movies["keywords"] +" "+ movies["overview"] +" "+ movies["tagline"] +" "+ movies["original_title"]
movies["title_length"] = movies["title"].apply(len)
movies["overview_length"] = movies["overview"].apply(len)
movies["runtime"] = movies["runtime"].fillna(0.0)

user_item_affinities = [[row[0], row[1], row[2]] for row in ratings.values]


from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding, FlairGlove100Embedding
from hwer import Feature, FeatureSet, ContentRecommendation, FeatureType
from hwer import HybridRecommender

embedding_mapper = {}
embedding_mapper['gender'] = CategoricalEmbedding(n_dims=1)
embedding_mapper['age'] = CategoricalEmbedding(n_dims=1)
embedding_mapper['occupation'] = CategoricalEmbedding(n_dims=2)
embedding_mapper['zip'] = CategoricalEmbedding(n_dims=2)

embedding_mapper['text'] = FlairGlove100Embedding()
embedding_mapper['numeric'] = NumericEmbedding(2)
embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=2)


recsys = ContentRecommendation(embedding_mapper=embedding_mapper, knn_params=None, n_output_dims=8, rating_scale=(1,5))


u1 = Feature(feature_name="gender", feature_type=FeatureType.CATEGORICAL, values=users.gender.values)
u2 = Feature(feature_name="age", feature_type=FeatureType.CATEGORICAL, values=users.age.astype(str).values)
u3 = Feature(feature_name="occupation", feature_type=FeatureType.CATEGORICAL, values=users.occupation.astype(str).values)
u4 = Feature(feature_name="zip", feature_type=FeatureType.CATEGORICAL, values=users.zip.astype(str).values)
user_data = FeatureSet([u1, u2, u3, u4])

i1 = Feature(feature_name="text", feature_type=FeatureType.STR, values=movies.text.values)
i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=movies.genres.values)
i3 = Feature(feature_name="numeric", feature_type=FeatureType.NUMERIC, values=movies[["title_length", "overview_length", "runtime"]].values)
item_data = FeatureSet([i1, i2, i3])

kwargs = {}
kwargs['user_data'] = user_data
kwargs['item_data'] = item_data


import warnings
warnings.filterwarnings('ignore')


hrecsys = HybridRecommender(embedding_mapper=embedding_mapper, knn_params=None, rating_scale=(1,5), n_content_dims=4, n_collaborative_dims=4)
user_vectors, item_vectors = hrecsys.fit(users.user_id.values, movies.movie_id.values,
               user_item_affinities, **kwargs)