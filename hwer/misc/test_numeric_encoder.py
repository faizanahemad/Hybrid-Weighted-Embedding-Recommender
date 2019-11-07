import sys
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
sys.path.append(path.join(path.dirname(__file__), '../'))

sys.path.insert(0, "../")

import sys
sys.path.append(os.getcwd())

import numpy as np
from hwer import CategoricalEmbedding, FeatureSet, Feature, NumericEmbedding, FeatureType


f1 = Feature("f1", FeatureType.NUMERIC, [1.2, 0.1, 2.2, 4.1, 5.0, 6.1, 2.1, 5.0])
f2 = Feature("f2", FeatureType.NUMERIC, [0.7, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 5.0])
fs = FeatureSet([f1, f2])

ns = NumericEmbedding(4)

print(ns.fit_transform(fs))
