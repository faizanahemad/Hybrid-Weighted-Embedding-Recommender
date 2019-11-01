from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
data = Dataset.load_builtin('ml-1m')
print(dir(data))
trainset, testset = train_test_split(data, test_size=.25)

print(len(data))

# Build a Fasttext Model
#