import sys
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
sys.path.append(path.join(path.dirname(__file__), '../'))

sys.path.insert(0, "../")

import sys
sys.path.append(os.getcwd())

import numpy as np
from hwer import MultiCategoricalEmbedding, FeatureSet, Feature, FeatureType

f1 = Feature("f1", FeatureType.MULTI_CATEGORICAL, [["a","b"],["b"],["c","b"],["a"],["a","c"],["a","b"],["b"]])
f2 = Feature("f2", FeatureType.NUMERIC, [1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 2.0])

fs = FeatureSet([f2])

cs = MultiCategoricalEmbedding(4, True)
p = cs.fit_transform(f1, target=fs)
print(p)

# without target
cs = MultiCategoricalEmbedding(4, True)
p = cs.fit_transform(f1)
print(p)





