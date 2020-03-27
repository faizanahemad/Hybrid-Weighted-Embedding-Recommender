import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from more_itertools import flatten

from .logging import getLogger
from .random_walk import *
from .svdpp_hybrid import SVDppHybrid
from .hybrid_graph_recommender import HybridGCNRec
from .utils import unit_length_violations
import logging
import dill
import sys
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class GCNRetriever(HybridGCNRec):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        assert n_collaborative_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        pass

    def __build_svd_model__(self, user_ids: List[str], item_ids: List[str],
                            user_item_affinities: List[Tuple[str, str, float]],
                            user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                            rating_scale: Tuple[float, float], **svd_params):
        pass

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        return [self.mu] * len(user_item_pairs)
