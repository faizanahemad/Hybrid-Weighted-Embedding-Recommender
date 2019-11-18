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

from hwer import FlairGlove100AndBytePairEmbedding,FlairGlove100Embedding, Feature, FeatureType

# text = list(map(lambda x: " ".join(x), common_texts))
#
# f1 = Feature("text", FeatureType.STR, text)
#
# flair1 = FlairGlove100Embedding()
# print(flair1.fit_transform(f1))
#
# flair2 = FlairGlove100AndBytePairEmbedding()
# print(flair2.fit_transform(f1))
#
# print(flair1.fit_transform(f1).shape)
# print(flair2.fit_transform(f1).shape)


#
f1 = Feature("text", FeatureType.STR, ["eifjcchchbnikfncbcntnhbvthnrbjiechcrbinucknb"])
flair1 = FlairGlove100Embedding()
print(flair1.fit_transform(f1))
