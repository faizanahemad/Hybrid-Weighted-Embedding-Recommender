import sys
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
sys.path.append(path.join(path.dirname(__file__), '../'))

sys.path.insert(0, "../")

import sys
sys.path.append(os.getcwd())

import numpy as np
from gensim.test.utils import common_texts

from hwer import FasttextEmbedding, Feature

ft = FasttextEmbedding(32, fasttext_file="/Users/ahemf/mygit/Hybrid-Weighted-Embedding-Recommender/hwer/fasttext.bin")

text = list(map(lambda x: " ".join(x), common_texts))
f1 = Feature("text", "str",str, text)
print(ft.fit_transform(f1))

print("="*40)
ft = FasttextEmbedding(4,)

f1 = Feature("text", "str",str, text)
print(ft.fit_transform(f1))

print(ft.fit_transform(f1).shape)