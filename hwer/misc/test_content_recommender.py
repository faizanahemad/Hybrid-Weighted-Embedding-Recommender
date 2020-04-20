import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hwer import ContentRecommendation
from hwer import NumericEmbed, Node, Edge

user_ids = ["A", "B", "C"]
items_per_user = 5
item_ids = list(map(str, range(items_per_user * len(user_ids))))

user_nodes = [Node("user", i) for i in user_ids]
item_nodes = [Node("item", i) for i in item_ids]

ia = np.random.normal(size=(items_per_user, 2))
ib = ia + 7
ib[:, 1] = ib[:, 1] - 3
ic = ia - 3
i1_15 = np.concatenate((ia, ib, ic))

edges = []
user_embeddings = []
for i, user in enumerate(user_ids):
    user_embeddings.append(np.average(i1_15[i * items_per_user:(i + 1) * items_per_user], axis=0, ))
    for j in range(i * items_per_user, (i + 1) * items_per_user):
        edges.append(Edge(Node("user", user), Node("item", item_ids[j]), 1))

print(edges)
print("=" * 60)

user_embeddings = np.vstack(user_embeddings)
actual_embeddings = np.concatenate((user_embeddings, i1_15))

embedding_mapper = {"item": {"numeric": NumericEmbed(n_dims=8)}}
node_data = {node: {"numeric": i1_15[idx]} for idx, node in enumerate(item_nodes)}

recsys = ContentRecommendation(embedding_mapper=embedding_mapper, n_dims=4, node_types={"user", "item"})
_ = recsys.fit(user_nodes + item_nodes, edges, node_data)

embeddings = recsys.get_embeddings(user_nodes + item_nodes)

all_entities = list(zip(user_ids, ["user"] * len(user_ids), [5] * len(user_ids))) + \
               list(zip(item_ids, ["item"] * len(item_ids), [1] * len(item_ids)))

all_nodes = user_nodes + item_nodes

print(len(embeddings), len(all_entities))

import operator

plt.figure(figsize=(8, 8))
sns.scatterplot(embeddings[:, 0], embeddings[:, 1], list(map(operator.itemgetter(1), all_entities)),
                size=list(map(operator.itemgetter(2), all_entities)))
plt.show()

import operator

plt.figure(figsize=(8, 8))
sns.scatterplot(actual_embeddings[:, 0], actual_embeddings[:, 1], list(map(operator.itemgetter(1), all_entities)),
                size=list(map(operator.itemgetter(2), all_entities)))
plt.show()

print(user_nodes[0], recsys.find_closest_neighbours("item", user_nodes[0], k=3))
