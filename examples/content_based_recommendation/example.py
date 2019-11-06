from surprise import Dataset
from surprise import accuracy
import pandas as pd
from pathlib import Path
from surprise.model_selection import train_test_split
data = Dataset.load_builtin('ml-1m')
print(dir(data))
trainset, testset = train_test_split(data, test_size=.25)

print(data.ratings_file)

path = Path(data.ratings_file)
ml_1m_dir = path.resolve().parents[1]
files = list(ml_1m_dir.glob('**/*.dat'))

users = [f for f in files if "users.dat" in str(f)][0]
movies = [f for f in files if "movies.dat" in str(f)][0]
ratings = [f for f in files if "ratings.dat" in str(f)][0]

users = pd.read_csv(str(users),sep="::", header=None, names=["user_id", "gender", "age", "occupation", "zip"], engine='python')
movies = pd.read_csv(str(movies),sep="::", header=None, names=["movie_id", "title", "genres"], engine='python')
ratings = pd.read_csv(str(ratings),sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine='python')

movies['genres'] = movies['genres'].apply(lambda x: x.lower().split('|'))
users['user_id'] = users['user_id'].astype(str)
movies['movie_id'] = movies['movie_id'].astype(str)
ratings['movie_id'] = ratings['movie_id'].astype(str)
ratings['user_id'] = ratings['user_id'].astype(str)
# CountVectorizer and make 1 column for each genre

print(users.shape, movies.shape, ratings.shape)

user_item_affinities = list(map(lambda x: tuple(x[0], x[1], x[2]), data.raw_ratings))



