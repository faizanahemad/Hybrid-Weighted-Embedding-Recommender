import abc
import operator
import time
from typing import List, Dict, Tuple, Optional, Set, Union

import numpy as np
import pandas as pd


from .content_recommender import ContentRecommendation
from .logging import getLogger
from .utils import unit_length, unit_length_violations
from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .embed import BaseEmbed

# Binary Link Prediction, Dot product based embeddings for KNN


class HybridRecommender(RecommendationBase):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]], node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(node_types=node_types,
                         n_dims=n_dims)
        self.cb = ContentRecommendation(embedding_mapper, node_types, np.inf)
        self.content_data_used = None
        self.log = getLogger(type(self).__name__)
        self.prediction_artifacts = dict()

    def __build_collaborative_embeddings__(self, nodes: List[Node],
                                           edges: List[Edge], node_vectors: np.ndarray,
                                           hyperparams: Dict) -> np.ndarray:

        self.log.debug("Started Building Collaborative Embeddings...")
        assert len(nodes) == len(node_vectors)
        collaborative_node_vectors = node_vectors
        assert np.sum(np.isnan(collaborative_node_vectors)) == 0
        return collaborative_node_vectors

    @abc.abstractmethod
    def __build_prediction_network__(self, nodes: List[Node],
                                     edges: List[Edge],
                                     content_vectors: np.ndarray, collaborative_vectors: np.ndarray,
                                     nodes_to_idx: Dict[Node, int],
                                     hyperparams: Dict):

        pass

    def fit(self,
            nodes: List[Node],
            edges: List[Edge],
            node_data: Dict[Node, Dict[FeatureName, object]],
            **kwargs):
        start_time = time.time()
        _ = super().fit(nodes, edges, node_data, **kwargs)
        self.log.debug("Hybrid Base: Fit Method Started")
        hyperparameters = {} if "hyperparameters" not in kwargs else kwargs["hyperparameters"]
        collaborative_params = {} if "collaborative_params" not in hyperparameters else hyperparameters["collaborative_params"]
        link_prediction_params = {} if "link_prediction_params" not in hyperparameters else \
            hyperparameters["link_prediction_params"]

        use_content = hyperparameters["use_content"] if "use_content" in hyperparameters else False
        content_data_used = len(node_data) != 0 and use_content
        self.content_data_used = content_data_used

        self.log.debug("Hybrid Base: Fit Method: content_data_used = %s", content_data_used)
        start = time.time()
        if content_data_used:
            super(type(self.cb), self.cb).fit(nodes, edges, node_data, **kwargs)
            content_vectors = self.cb.__build_content_embeddings__(nodes, edges, node_data, np.inf)
            self.cb = None
            del self.cb

        else:
            content_vectors = np.random.rand(len(nodes), 1)
        self.log.info("Hybrid Base: Built Content Embedding., shape = %s, Time = %.1f" %
                       (content_vectors.shape, time.time() - start))
        import gc
        gc.collect()

        collaborative_vectors = self.__build_collaborative_embeddings__(nodes, edges, content_vectors, collaborative_params)

        self.log.debug("Hybrid Base: Fit Method, Use content = %s, Unit Length Violations: %s", content_data_used,
                       unit_length_violations(collaborative_vectors, axis=1))
        collaborative_vectors = unit_length(collaborative_vectors, axis=1)
        gc.collect()

        # assert collaborative_vectors.shape[1] == self.n_dims
        self.log.debug("Hybrid Base: Start Building Prediction Network...")
        prediction_artifacts = self.__build_prediction_network__(nodes, edges,
                                                                 content_vectors, collaborative_vectors,
                                                                 self.nodes_to_idx, link_prediction_params)

        if prediction_artifacts is not None:
            self.prediction_artifacts.update(dict(prediction_artifacts))
        gc.collect()
        self.log.debug("Hybrid Base: Built Prediction Network.")

        knn_vectors = self.prepare_for_knn(content_vectors, collaborative_vectors)
        self.__build_knn__(knn_vectors)
        self.fit_done = True
        self.log.info("End Fitting Recommender, vectors shape = %s, Time to fit = %.1f",
                      self.vectors.shape, time.time() - start_time)
        gc.collect()
        return self.vectors

    @abc.abstractmethod
    def prepare_for_knn(self, content_vectors: np.ndarray, collaborative_vectors: np.ndarray) -> np.ndarray:
        from .utils import unit_length
        from sklearn.decomposition import PCA
        if collaborative_vectors.shape[1] > self.n_dims:
            pca = PCA(n_components=self.n_dims)
            collaborative_vectors = pca.fit_transform(collaborative_vectors)
        elif collaborative_vectors.shape[1] < self.n_dims:
            raise ValueError()
        collaborative_vectors = unit_length(collaborative_vectors, axis=1)
        return collaborative_vectors
