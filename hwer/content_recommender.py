from typing import List, Dict, Set

import numpy as np
from bidict import bidict
from sklearn.preprocessing import OneHotEncoder

from .embed import BaseEmbed
from .logging import getLogger
from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .utils import unit_length, auto_encoder_transform


class ContentRecommendation(RecommendationBase):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]],
                 node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(node_types=node_types,
                         n_dims=n_dims)

        self.embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]] = embedding_mapper
        self.log = getLogger(type(self).__name__)

    def __build_content_embeddings__(self, nodes: List[Node], edges: List[Edge],
                                     node_data: Dict[Node, Dict[FeatureName, object]], n_dims):
        self.log.debug("ContentRecommendation::__build_embeddings__:: Started...")
        all_embeddings = None
        node_to_idx_internal = bidict()
        for nt in self.node_types:
            nt_embedding = None
            nt_nodes = list(filter(lambda n: n.node_type == nt, nodes))
            assert len(set(nt_nodes) - set(node_data.keys())) == 0 or len(set(nt_nodes) - set(node_data.keys())) == len(
                set(nt_nodes))
            assert len(set(nt_nodes)) == len(nt_nodes)
            if len(set(nt_nodes) - set(node_data.keys())) == len(set(nt_nodes)):
                nt_embedding = np.zeros((len(nt_nodes), 1))
            else:
                nt_nodes_features: List[Dict[FeatureName, object]] = [node_data[ntn] for ntn in nt_nodes]
                feature_names = list(nt_nodes_features[0].keys())

                for f in feature_names:
                    feature = [ntnf[f] for ntnf in nt_nodes_features]
                    embedding = self.embedding_mapper[nt][f].fit_transform(feature)
                    if nt_embedding is None:
                        nt_embedding = embedding
                    else:
                        np.concatenate((nt_embedding, embedding), axis=1)
                nt_embedding = unit_length(nt_embedding, axis=1)

            #
            cur_len = len(node_to_idx_internal)
            node_to_idx_internal.update(bidict(zip(nt_nodes, range(cur_len, cur_len + len(nt_nodes)))))
            if all_embeddings is None:
                all_embeddings = nt_embedding
            else:
                c1 = np.concatenate((all_embeddings, np.zeros((all_embeddings.shape[0], nt_embedding.shape[1]))),
                                    axis=1)
                c2 = np.concatenate((np.zeros((nt_embedding.shape[0], all_embeddings.shape[1])), nt_embedding), axis=1)
                all_embeddings = np.concatenate((c1, c2), axis=0)

        all_embeddings = all_embeddings[[node_to_idx_internal[n] for n in nodes]]
        nts = np.array([n.node_type for n in nodes]).reshape((-1, 1))
        ohe_node_types = OneHotEncoder(sparse=False).fit_transform(nts)
        all_embeddings = np.concatenate((all_embeddings, ohe_node_types), axis=1)
        self.log.debug(
            "ContentRecommendation::__build_embeddings__:: AutoEncoder with dims = %s" % str(all_embeddings.shape))
        n_dims = n_dims if n_dims is not None and not np.isinf(n_dims) else 2 ** int(np.log2(all_embeddings.shape[1]))
        all_embeddings, _ = auto_encoder_transform(all_embeddings, all_embeddings, n_dims=n_dims, verbose=2, epochs=25)
        all_embeddings = unit_length(all_embeddings, axis=1)
        self.log.info("ContentRecommendation::__build_embeddings__:: Built Content Embedding with dims = %s" % str(
            all_embeddings.shape))
        return all_embeddings

    def fit(self,
            nodes: List[Node],
            edges: List[Edge],
            node_data: Dict[Node, Dict[FeatureName, object]],
            **kwargs):

        super().fit(nodes, edges, node_data)
        embeddings = self.__build_content_embeddings__(nodes, edges, node_data, self.n_dims)
        self.__build_knn__(embeddings)

        # AutoEncoder them so that error is minimised and distance is maintained
        # https://stats.stackexchange.com/questions/351212/do-autoencoders-preserve-distances
        # Distance Preserving vs Non Preserving

        self.fit_done = True
        return embeddings
