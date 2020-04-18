
from hwer import MultiCategoricalEmbedding, FlairGlove100AndBytePairEmbedding, CategoricalEmbedding, NumericEmbedding
from hwer import Feature, FeatureSet, ContentRecommendation, FeatureType, EntityType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

user_ids = ["A", "B", "C"]
items_per_user = 50
item_ids = list(map(str, range(items_per_user * 3)))

ia = np.random.normal(size=(items_per_user, 2))
ib = ia + 7
ib[:, 1] = ib[:, 1] - 3
ic = ia - 3
i1_15 = np.concatenate((ia, ib, ic))

user_item_affinities = []
user_embeddings = []
for i, user in enumerate(user_ids):
    user_embeddings.append(np.average(i1_15[i*items_per_user:(i+1)*items_per_user], axis=0,))
    for j in range(i*items_per_user, (i+1)*items_per_user):
        user_item_affinities.append((user, item_ids[j], 3))


user_embeddings = np.vstack(user_embeddings)
actual_embeddings = np.concatenate((user_embeddings, i1_15))

embedding_mapper = {}
embedding_mapper['numeric'] = NumericEmbedding(n_dims=3)

f = Feature("numeric", FeatureType.NUMERIC, i1_15)
item_data = FeatureSet([f])

kwargs = {'item_data': item_data}

recsys = ContentRecommendation(embedding_mapper=embedding_mapper, knn_params=None, n_dims=2, rating_scale=(1, 5))
_ = recsys.fit(user_ids, item_ids,
               user_item_affinities, **kwargs)

all_entities = list(zip(user_ids, [EntityType.USER]*len(user_ids))) +\
               list(zip(item_ids, [EntityType.ITEM]*len(item_ids)))

embeddings = recsys.get_embeddings(all_entities)

all_entities = list(zip(user_ids, [EntityType.USER]*len(user_ids), [5]*len(user_ids))) +\
               list(zip(item_ids, [EntityType.ITEM]*len(item_ids), [1]*len(item_ids)))

print(len(embeddings), len(all_entities))

import operator
plt.figure(figsize=(8, 8))
sns.scatterplot(embeddings[:,0],embeddings[:,1], list(map(operator.itemgetter(1), all_entities)), size=list(map(operator.itemgetter(2), all_entities)))
plt.show()

import operator
plt.figure(figsize=(8, 8))
sns.scatterplot(actual_embeddings[:,0],actual_embeddings[:,1], list(map(operator.itemgetter(1), all_entities)), size=list(map(operator.itemgetter(2), all_entities)))
plt.show()






