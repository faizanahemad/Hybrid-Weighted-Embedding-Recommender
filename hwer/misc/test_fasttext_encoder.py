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
from hwer.utils import cos_sim

from hwer import FastTextEmbed


print("X"*80)
f1 = [["the cat sat on the mat"], ["mat is the cat's place"]]
flair1 = FastTextEmbed(4)
p = flair1.fit_transform(f1)
# print(p)
examples = len(f1)

print("-" * 40)
sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
print(p.shape)


print("-X-"*80)
f1 = [["the cat sat on the mat", "Purple pie with a cricket bat"], ["mat is the cat's place"]]
flair1 = FastTextEmbed(4)
p = flair1.fit_transform(f1)
# print(p)
examples = len(f1)

print("-" * 40)
sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
print(p.shape)

print("=X="*80)
f1 = [["the cat sat on the mat", "the cat sat on the mat"], ["the cat sat on the mat"]]
flair1 = FastTextEmbed(4)
p = flair1.fit_transform(f1)
# print(p)
examples = len(f1)

print("-" * 40)
sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
print(p.shape)

print("<y>"*80)
f1 = ["the cat sat on the mat", ["the cat sat on the mat"]]
flair1 = FastTextEmbed(4)
p = flair1.fit_transform(f1)
# print(p)
examples = len(f1)

print("-" * 40)
sims = np.zeros((examples, examples))
for i in range(examples):
    for j in range(examples):
        sims[i][j] = cos_sim(p[i], p[j])
print(sims)
print(p.shape)


