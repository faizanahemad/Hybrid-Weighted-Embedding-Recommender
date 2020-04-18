import abc
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set, Union
import collections

import numpy as np
from bidict import bidict

from .logging import getLogger
from .utils import is_num, is_2d_array, NodeNotFoundException
from .utils import normalize_affinity_scores_by_user_item_bs, unit_length
import operator
from sklearn.neighbors import KDTree


class FeatureType(Enum):
    ID = 1
    NUMERIC = 2
    STR = 3
    CATEGORICAL = 4
    MULTI_CATEGORICAL = 5


class Feature:
    def __init__(self, feature_name: str,
                 feature_type: FeatureType,
                 values: List):
        """

        :param feature_name:
        :param feature_type:
        :param values:
        """
        self.feature_name: str = feature_name
        self.feature_type: FeatureType = feature_type
        assert type(feature_type) == FeatureType
        self.values = values
        if feature_type is FeatureType.ID:
            assert type(self.values[0]) == str or type(self.values[0]) == int
        if feature_type is FeatureType.NUMERIC:
            assert is_num(self.values[0]) or (is_2d_array(self.values) and is_num(self.values[0][0]))
        if feature_type is FeatureType.STR:
            assert isinstance(self.values[0], str)
        if feature_type is FeatureType.CATEGORICAL:
            assert isinstance(self.values[0], str) or (is_2d_array(self.values) and isinstance(self.values[0][0], str))
        if feature_type is FeatureType.MULTI_CATEGORICAL:
            assert is_2d_array(self.values)

    def __len__(self):
        return len(self.values)


class FeatureSet:
    def __init__(self, features: List[Feature]):
        self.features = features
        self.feature_names = [f.feature_name for f in features]
        self.feature_types = [f.feature_type for f in features]
        assert self.feature_types.count(FeatureType.ID) <= 1
        # check all features have same count of values in FeatureSet
        assert len(set([len(f) for f in features])) <= 1

    def __getitem__(self, key):
        return self.features[key]


NodeType = str
NodeExternalId = Union[str, int]

Node = collections.namedtuple("Node", ["node_type", "node_external_id"])
Edge = collections.namedtuple("Edge", ["src", "dst", "weight"])


class MultiKNN:
    def __init__(self, nodes_to_idx: bidict[Node, int], vectors: np.ndarray, leaf_size=128,):
        self.nodes_to_idx = nodes_to_idx
        assert len(nodes_to_idx) == len(vectors)
        vl: Dict[str, List[int]] = defaultdict(list)
        for n, i in nodes_to_idx.items():
            nt = n.node_type
            vl[nt].append(i)

        idxs = {k: bidict(zip(range(len(v)), v)) for k, v in vl.items()}
        knn = {k: KDTree(vectors[v], leaf_size=leaf_size) for k, v in vl.items()}
        self.idxs = idxs
        self.knn = knn

    def query(self, embedding, node_type, k=200) -> List[Tuple[Node, float]]:
        (dist,), (neighbors,) = self.knn[node_type].query([embedding], k=k)
        neighbors = [self.idxs[n] for n in neighbors]
        results = [(self.nodes_to_idx.inverse[idx], dt) for idx, dt in zip(neighbors, dist)]
        results = list(sorted(results, key=operator.itemgetter(1), reverse=False))
        return results


class RecommendationBase(metaclass=abc.ABCMeta):
    def __init__(self, node_types: Set[str], n_dims: int = 32):
        self.node_types: Set[NodeType] = node_types
        self.nodes_to_idx: bidict[Node, int] = None
        self.knn: MultiKNN = None
        self.vectors: np.ndarray = None
        self.fit_done = False
        self.n_dims = n_dims
        self.log = getLogger(type(self).__name__)

    def add_nodes(self, nodes: List[Node]):
        assert len(set(nodes)) == len(nodes)
        assert self.nodes_to_idx.keys().isdisjoint(set(nodes))
        assert len(set([n.node_type for n in nodes]) - self.node_types) == 0
        all_count = len(self.nodes_to_idx)
        global_update = bidict(zip(nodes, list(range(all_count, all_count + len(nodes)))))
        self.nodes_to_idx.update(global_update)
        return self

    def __build_knn__(self, vectors: np.ndarray):
        self.knn = MultiKNN(self.nodes_to_idx, vectors, leaf_size=128)
        return self

    # Convert edges into global id based edges
    @abc.abstractmethod
    def fit(self,
            nodes: List[Node],
            edges: List[Edge],
            **kwargs):
        # self.build_content_embeddings(item_data, user_item_affinities)
        assert not self.fit_done
        sparsity = 1 - len(edges) / (len(nodes) * len(nodes))
        self.log.info("Start Fitting Base Recommender with nodes = %s, edges = %s, sparsity = %s",
                      len(nodes), len(edges), sparsity)

        # Check if all edges are made by nodes in node list
        assert len(set([i for e in edges for i in [e.src, e.dst]]) - set(nodes)) == 0
        assert len(set(nodes)) == len(nodes)
        self.add_nodes(nodes)
        self.log.info("End Fitting Base Recommender")
        return edges

    @abc.abstractmethod
    def predict(self, node_pairs: List[Tuple[Node, Node]]) -> List[float]:
        """

        :param node_pairs: Takes a list of pair of nodes and gives proba of a link existing between them.
        :return: Probabilities of link existence between respective node pairs
        """
        pass

    def get_embeddings(self, nodes: List[Node]):
        indexes = [self.nodes_to_idx[node] for node in nodes]
        embeddings = self.vectors[indexes]
        return embeddings

    def get_average_embeddings(self, entities: List[Node]):
        embeddings = self.get_embeddings(entities)
        return  unit_length(np.average(embeddings, axis=0))

    def find_closest_neighbours(self, node_type: str, anchor: Node, positive: List[Node] = None,
                                negative: List[Node] = None, k=200) -> List[Tuple[Node, float]]:
        assert self.fit_done
        assert node_type in self.node_types and node_type in self.knn
        if anchor not in self.nodes_to_idx:
            raise NodeNotFoundException("Node = %s, was not provided in training" % anchor)

        embedding_list = [self.get_average_embeddings([Node])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)
        return self.knn.query(embedding, node_type, k=k)

