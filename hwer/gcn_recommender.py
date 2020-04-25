import time
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

from .logging import getLogger
from .random_walk import *
from .gcn_ncf import GcnNCF
from .utils import NodeNotFoundException
import operator
from .utils import unit_length_violations
import logging
import dill
import sys
from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .embed import BaseEmbed
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class GCNRecommender(GcnNCF):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]], node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(embedding_mapper, node_types, n_dims)
        self.log = getLogger(type(self).__name__)
        assert n_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

    def __build_prediction_network__(self, nodes: List[Node],
                                     edges: List[Edge],
                                     content_vectors: np.ndarray, collaborative_vectors: np.ndarray,
                                     nodes_to_idx: Dict[Node, int],
                                     hyperparams: Dict):

        pass

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

        # 2 to 0 => 0 to 1
        # -1*x + 2 -> -2 to 0 -> 0 to 2 -> x/2
        # (-1*x + 2)/2
        embedding = np.average(embedding_list, axis=0)
        node_dist_list = self.knn.query(embedding, node_type, k=k)
        nodes, dist = zip(*node_dist_list)
        dist = np.array(dist)
        dist = (-1 * dist + 2)/2
        node_dist_list = zip(nodes, dist)
        results = list(sorted(node_dist_list, key=operator.itemgetter(1), reverse=True))
        return results
