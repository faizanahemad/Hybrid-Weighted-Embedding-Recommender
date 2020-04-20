import sys
import os
from os import path
from sklearn.preprocessing import MinMaxScaler
sys.path.append(path.join(path.dirname(__file__), '../'))

sys.path.insert(0, "../")

import sys
sys.path.append(os.getcwd())

import numpy as np

from hwer import NumericEmbed
from hwer.utils import cos_sim

vals = np.random.random((6, 3))
examples = len(vals)

sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(vals[i], vals[j])
print(sims)

print("=" * 80)

p = NumericEmbed(n_dims=3).fit_transform(vals)
print(p)

print("=" * 80)
sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
