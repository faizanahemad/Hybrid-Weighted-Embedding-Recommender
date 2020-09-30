import abc
import operator
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Union

import numpy as np
from bidict import bidict
from sklearn.neighbors import KDTree

from .logging import getLogger
from .utils import NodeNotFoundException
from .utils import unit_length, unit_length_violations

NodeType = str
NodeExternalId = Union[str, int]
FeatureName = str


class Node:
    def __init__(self, node_type, node_external_id):
        self.node_type = node_type
        self.node_external_id = str(node_external_id)

    def __key(self):
        return tuple((self.node_type, self.node_external_id))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return str(self.__key())


class Edge:
    def __init__(self, src: Node, dst: Node, weight: float):
        self.src = src
        self.dst = dst
        self.weight = weight
        self.contents = [src, dst, weight]

    def __key(self):
        return tuple((self.src, self.dst, self.weight))

    def __iter__(self):
        return iter(self.contents)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return "{src: %s, dst: %s, weight: %s}" % (self.src, self.dst, self.weight)


class MultiKNN:
    def __init__(self, nodes_to_idx: Dict[Node, int], vectors: np.ndarray, leaf_size=128, ):
        self.nodes_to_idx: bidict = nodes_to_idx
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
        neighbors = [self.idxs[node_type][n] for n in neighbors]
        results = [(self.nodes_to_idx.inverse[idx], dt) for idx, dt in zip(neighbors, dist)]
        results = list(sorted(results, key=operator.itemgetter(1), reverse=False))
        return results


class RecommendationBase(metaclass=abc.ABCMeta):
    def __init__(self, node_types: Set[str], n_dims: int = 32):
        self.node_types: Set[NodeType] = node_types
        self.nodes_to_idx: bidict = bidict()
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
        v, _, _, _ = unit_length_violations(vectors, axis=1)
        assert v == 0
        self.knn = MultiKNN(self.nodes_to_idx, vectors, leaf_size=128)
        self.vectors = vectors
        return self

    @abc.abstractmethod
    def fit(self,
            nodes: List[Node],
            edges: List[Edge],
            node_data: Dict[Node, Dict[FeatureName, object]],
            **kwargs):
        # self.build_content_embeddings(item_data, user_item_affinities)
        assert not self.fit_done
        sparsity = 1 - len(edges) / (len(nodes) * len(nodes))
        self.log.info("Start Fitting Base Recommender with nodes = %s, edges = %s, sparsity = %s",
                      len(nodes), len(edges), sparsity)

        # Check if all edges are made by nodes in node list
        edge_node_types = set([node.node_type for e in edges for node in [e.src, e.dst]])
        print("Edge node types = ", edge_node_types, "Actual Node types = ", self.node_types)
        assert edge_node_types == self.node_types
        assert len(set([i for e in edges for i in [e.src, e.dst]]) - set(nodes)) == 0

        node_type_set_nodes = defaultdict(set)
        for n in nodes:
            node_type_set_nodes[n.node_type].add(n)

        node_type_set_edges = defaultdict(set)
        for e in edges:
            node_type_set_edges[e.src.node_type].add(e.src)
            node_type_set_edges[e.dst.node_type].add(e.dst)

        print(node_type_set_nodes.keys(), node_type_set_edges.keys())
        for k, v in node_type_set_nodes.items():
            if not len(node_type_set_edges[k] - v) == 0:
                print(k, "failed", "# nodes = %s, # from edges = %s" %(len(v), len(node_type_set_edges[k])))

        assert len(set(nodes)) == len(nodes)
        assert len(set([n.node_type for n in nodes]) - self.node_types) == 0
        self.add_nodes(nodes)
        self.log.info("End Fitting Base Recommender")
        return edges

    def predict(self, node_pairs: List[Tuple[Node, Node]]) -> List[float]:
        """

        :param node_pairs: Takes a list of pair of nodes and gives proba of a link existing between them.
        :return: Probabilities of link existence between respective node pairs
        """
        src, dst = zip(*node_pairs)
        results = (self.get_embeddings(src) * self.get_embeddings(dst)).sum(1)
        results = (results + 1) / 2
        return results

    def get_embeddings(self, nodes: List[Node]):
        indexes = np.array([self.nodes_to_idx[node] if node in self.nodes_to_idx else -1 for node in nodes])
        mask = indexes == -1
        embeddings = self.vectors[np.where(indexes >= 0, indexes, 0)]
        embeddings[mask] = np.clip(embeddings[mask], 1e-6, 1e-5)
        return embeddings

    def get_average_embeddings(self, entities: List[Node]):
        embeddings = self.get_embeddings(entities)
        return unit_length(np.average(embeddings, axis=0))

    def find_closest_neighbours(self, node_type: str, anchor: Node, positive: List[Node] = None,
                                negative: List[Node] = None, k=200) -> List[Tuple[Node, float]]:
        assert self.fit_done
        assert node_type in self.node_types and node_type in self.knn.knn
        if anchor not in self.nodes_to_idx:
            raise NodeNotFoundException("Node = %s, was not provided in training" % anchor)

        embedding_list = [self.get_average_embeddings([anchor])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)
        node_dist_list = self.knn.query(embedding, node_type, k=k)
        scores = self.predict([(anchor, node) for node, dist in node_dist_list])
        results = list(sorted(zip([n for n, d in node_dist_list], scores), key=operator.itemgetter(1), reverse=True))
        return results
