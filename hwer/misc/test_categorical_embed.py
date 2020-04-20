import sys
import os
from os import path

import sys
sys.path.append(os.getcwd())

import numpy as np
from hwer import CategoricalEmbed
from hwer.utils import cos_sim

f1 = [["a", "b", 1, ["ab", "ca"], "c"],
      ["e", "a", 2, ["ab", "bc"], "b"],
      ["b", "b", 2, ["ab", "ab"], "b"],
      ["a", "b", 2, ["ca", "ab"], "a"],
      ["c", "a", 2, ["ab", "ab"], "b"],
      ["a", "b", 1, ["ab", "ca"], "c"],
      ["a", "b", 2, ["ab", "ca"], "c"]]


cs = CategoricalEmbed(8, True)
p = cs.fit_transform(f1)
print(p)

print("=" * 80)
sims = np.zeros((len(f1), len(f1)))
for i in range(len(f1)):
    for j in range(len(f1)):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
