from surprise import Dataset
from surprise import accuracy
import pandas as pd
from pathlib import Path
from surprise.model_selection import train_test_split
data = Dataset.load_builtin('ml-1m')
trainset, testset = train_test_split(data, test_size=.25)

print(data.ratings_file)

path = Path(data.ratings_file)
ml_1m_dir = path.resolve().parents[1]
files = list(ml_1m_dir.glob('**/*.dat'))

users = [f for f in files if "users.dat" in str(f)][0]
movies = [f for f in files if "movies.dat" in str(f)][0]
ratings = [f for f in files if "ratings.dat" in str(f)][0]

users = pd.read_csv(str(users), sep="::", header=None, names=["user_id", "gender", "age", "occupation", "zip"], engine='python')
movies = pd.read_csv(str(movies), sep="::", header=None, names=["movie_id", "title", "genres"], engine='python')
ratings = pd.read_csv(str(ratings), sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine='python')

movies['genres'] = movies['genres'].apply(lambda x: x.lower().split('|'))
users['user_id'] = users['user_id'].astype(str)
movies['movie_id'] = movies['movie_id'].astype(str)
ratings['movie_id'] = ratings['movie_id'].astype(str)
ratings['user_id'] = ratings['user_id'].astype(str)
# CountVectorizer and make 1 column for each genre

print(users.shape, movies.shape, ratings.shape)

user_item_affinities = list(map(lambda x: tuple([x[0], x[1], x[2]]), data.raw_ratings))

from importlib import reload
import hwer
reload(hwer)

from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding
from hwer import Feature, FeatureSet, ContentRecommendation, FeatureType

embedding_mapper = {}
embedding_mapper['gender'] = CategoricalEmbedding(n_dims=2)
embedding_mapper['age'] = CategoricalEmbedding(n_dims=2)
embedding_mapper['occupation'] = CategoricalEmbedding(n_dims=2)
embedding_mapper['zip'] = CategoricalEmbedding(n_dims=8)

embedding_mapper['title'] = FlairGlove100AndBytePairEmbedding()
embedding_mapper['genres'] = MultiCategoricalEmbedding(n_dims=16)


recsys = ContentRecommendation(embedding_mapper=embedding_mapper, knn_params=None, n_output_dims=64)

kwargs = {'user_item_affinities':user_item_affinities}

u1 = Feature(feature_name="gender", feature_type=FeatureType.CATEGORICAL, values=users.gender.values)
u2 = Feature(feature_name="age", feature_type=FeatureType.CATEGORICAL, values=users.age.astype(str).values)
u3 = Feature(feature_name="occupation", feature_type=FeatureType.CATEGORICAL, values=users.occupation.astype(str).values)
u4 = Feature(feature_name="zip", feature_type=FeatureType.CATEGORICAL, values=users.zip.astype(str).values)
user_data = FeatureSet([u1, u2, u3, u4])

i1 = Feature(feature_name="title", feature_type=FeatureType.STR, values=movies.title.values)
i2 = Feature(feature_name="genres", feature_type=FeatureType.MULTI_CATEGORICAL, values=movies.genres.values)
item_data = FeatureSet([i1, i2])

kwargs['user_data'] = user_data
kwargs['item_data'] = item_data

recsys.fit(user_ids=users.user_id.values, item_ids=movies.movie_id.values, **kwargs)

res, dist = zip(*recsys.find_items_for_user(user='1', positive=[], negative=[]))
res = res[:20]


