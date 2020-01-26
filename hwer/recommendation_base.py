import abc
from collections import defaultdict
from enum import Enum
from typing import List, Tuple, Optional

import hnswlib
import numpy as np
from bidict import bidict

from .logging import getLogger
from .utils import is_num, is_2d_array, UserNotFoundException, ItemNotFoundException
from .utils import normalize_affinity_scores_by_user_item_bs, unit_length
import operator


# TODO: Add Validations for add apis
# TODO: Use stable sort for item ranking
# TODO: Don't store the data just use it in embedding
# TODO: For Above provide cleanup/delete APIs
# TODO: Support Categorical features

class EntityType(Enum):
    USER = 1
    ITEM = 2
    USER_ITEM = 3


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


class RecommendationBase(metaclass=abc.ABCMeta):
    def __init__(self, knn_params: dict, rating_scale: Tuple[float, float],
                 n_output_dims: int = 32):
        self.users_set = set()
        self.items_set = set()

        self.user_features = None
        self.item_features = None
        self.item_only_features = None
        self.user_only_features = None

        self.user_id_to_index = bidict()
        self.item_id_to_index = bidict()
        self.user_knn = None
        self.item_knn = None
        self.fit_done = False
        self.knn_params = knn_params
        if self.knn_params is None:
            self.knn_params = dict(n_neighbors=100,
                                   index_time_params={'M': 15, 'ef_construction': 200, })

        self.n_output_dims = n_output_dims
        self.rating_scale = rating_scale
        self.mu: Optional[int] = None
        self.bu = defaultdict(float)
        self.bi = defaultdict(float)
        self.spread: Optional[int] = None
        self.log = getLogger(type(self).__name__)

    def __add_users__(self, users: List[str]):
        new_users_set = set(users)
        if len(new_users_set.intersection(self.users_set)) > 0:
            raise AssertionError("Trying to Add User ID already present.")
        current_user_count = len(self.users_set)
        update_dict = bidict(zip(users, list(range(current_user_count, current_user_count + len(users)))))
        self.user_id_to_index.update(update_dict)
        self.users_set.update(new_users_set)
        return self

    def __add_items__(self, items: List[str]):
        new_items_set = set(items)
        if len(new_items_set.intersection(self.items_set)) > 0:
            raise AssertionError("Trying to Add Item ID already present.")

        current_item_count = len(self.items_set)
        update_dict = bidict(zip(items, list(range(current_item_count, current_item_count + len(items)))))
        self.item_id_to_index.update(update_dict)
        self.items_set.update(new_items_set)
        return self

    def __build_knn__(self, user_ids: List[str], item_ids: List[str],
                      user_vectors: np.ndarray, item_vectors: np.ndarray):

        n_neighbors = self.knn_params["n_neighbors"]
        index_time_params = self.knn_params["index_time_params"]
        user_vectors = unit_length(user_vectors, axis=1)
        item_vectors = unit_length(item_vectors, axis=1)
        user_knn = hnswlib.Index(space='cosine', dim=user_vectors.shape[1])
        user_knn.init_index(max_elements=len(user_ids) * 2,
                            ef_construction=index_time_params['ef_construction'], M=index_time_params['M'])
        user_knn.set_ef(n_neighbors * 2)
        assert len(user_vectors) == len(self.users_set)
        assert user_vectors.shape[1] == item_vectors.shape[1]
        user_knn.add_items(user_vectors, list(range(len(self.users_set))))

        item_knn = hnswlib.Index(space='cosine', dim=item_vectors.shape[1])
        item_knn.init_index(max_elements=len(item_ids) * 2,
                            ef_construction=index_time_params['ef_construction'], M=index_time_params['M'])
        item_knn.set_ef(n_neighbors * 2)
        assert len(item_vectors) == len(self.items_set)
        item_knn.add_items(item_vectors, list(range(len(self.items_set))))

        self.user_knn = user_knn
        self.item_knn = item_knn
        self.log.info("Built KNN, user vectors shape = %s, item vectors shape = %s, n_neighbors = %s",
                      user_vectors.shape,
                      item_vectors.shape, n_neighbors)
        return user_knn, item_knn

    def add_user(self, user_id: str, features: FeatureSet, user_item_affinities: List[Tuple[str, str, float]]):
        assert self.fit_done
        self.__add_users__(users=[user_id])
        assert set([f.feature_name for f in features if f.feature_type != FeatureType.ID]) == set(self.user_features)
        users, items, weights = zip(*user_item_affinities)
        embedding = np.average(self.get_average_embeddings(items), weights=weights)
        self.user_knn.add_items([embedding], [len(self.users_set) - 1])

    def add_item(self, item_id, features: FeatureSet, user_item_affinities: List[Tuple[str, str, float]]):
        assert self.fit_done
        self.__add_items__(items=[item_id])
        assert set([f.feature_name for f in features if f.feature_type != FeatureType.ID]) == set(self.item_features)
        users, items, weights = zip(*user_item_affinities)
        embedding = np.average(self.get_average_embeddings(users), weights=weights)
        self.item_knn.add_items([embedding], [len(self.items_set) - 1])

    @abc.abstractmethod
    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_item_affinities: List[Tuple[str, str, float]],
            **kwargs):
        # self.build_content_embeddings(item_data, user_item_affinities)
        assert not self.fit_done
        sparsity = 1 - len(user_item_affinities) / (len(user_ids) * len(item_ids))
        self.log.info("Start Fitting Recommender with n_users = %s, n_items = %s, n_samples = %s, sparsity = %s",
                      len(user_ids), len(item_ids), len(user_item_affinities), sparsity)
        user_set = set(user_ids)
        item_set = set(item_ids)
        assert len(user_ids) == len(user_set)
        assert len(item_ids) == len(item_set)
        assert len(list([(u, i, r) for u, i, r in user_item_affinities if u not in user_set or i not in item_set])) == 0
        self.__add_users__(user_ids)
        self.__add_items__(item_ids)

        item_data: FeatureSet = kwargs["item_data"] if "item_data" in kwargs else FeatureSet([])
        user_data: FeatureSet = kwargs["user_data"] if "user_data" in kwargs else FeatureSet([])
        self.user_features = [feature.feature_name for feature in user_data if feature.feature_type != FeatureType.ID]
        self.item_features = [feature.feature_name for feature in item_data if feature.feature_type != FeatureType.ID]
        self.item_only_features = list(set(self.item_features) - set(self.user_features))
        self.user_only_features = list(set(self.user_features) - set(self.item_features))
        self.mu, self.bu, self.bi, self.spread, uid = normalize_affinity_scores_by_user_item_bs(user_item_affinities)
        return uid

    @abc.abstractmethod
    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        pass

    def get_embeddings(self, entities: List[Tuple[str, EntityType]]):
        users = list(filter(lambda x: x[1] == EntityType.USER, entities))
        items = list(filter(lambda x: x[1] == EntityType.ITEM, entities))
        user_vectors = None
        item_vectors = None
        if len(users) > 0:
            users, _ = zip(*users)
            users = [self.user_id_to_index[u] for u in users]
            user_vectors = self.user_knn.get_items(users)
        if len(items) > 0:
            items, _ = zip(*items)
            items = [self.item_id_to_index[i] for i in items]
            item_vectors = self.item_knn.get_items(items)
        if user_vectors is not None and item_vectors is not None:
            embeddings = np.concatenate((user_vectors, item_vectors), axis=0)
        elif user_vectors is not None:
            embeddings = user_vectors
        elif item_vectors is not None:
            embeddings = item_vectors
        else:
            raise ValueError("No Embeddings Found")
        return embeddings

    def get_average_embeddings(self, entities: List[Tuple[str, EntityType]]):
        embeddings = self.get_embeddings(entities)
        return np.average(embeddings, axis=0)

    def find_similar_items(self, item: str, positive: List[Tuple[str, EntityType]] = None,
                           negative: List[Tuple[str, EntityType]] = None, k=None) \
            -> List[Tuple[str, float]]:
        assert self.fit_done
        if item not in self.items_set:
            raise ItemNotFoundException("Item with itemID = %s, was not provided in training" % item)
        k = self.knn_params['n_neighbors'] if k is None else k
        embedding_list = [self.get_average_embeddings([(item, EntityType.ITEM)])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)

        (neighbors,), (dist,) = self.item_knn.knn_query([embedding], k=k)
        results = [(self.item_id_to_index.inverse[idx], dt) for idx, dt in zip(neighbors, dist)]
        return list(sorted(results, key=operator.itemgetter(1), reverse=False))

    def find_similar_users(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                           negative: List[Tuple[str, EntityType]] = None, k=None) -> List[Tuple[str, float]]:
        assert self.fit_done
        if user not in self.users_set:
            raise UserNotFoundException("User with userID = %s, was not provided in training" % user)
        k = self.knn_params['n_neighbors'] if k is None else k
        embedding_list = [self.get_average_embeddings([(user, EntityType.USER)])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)
        (neighbors,), (dist,) = self.user_knn.knn_query([embedding], k=k)
        results = [(self.user_id_to_index.inverse[idx], dt) for idx, dt in zip(neighbors, dist)]
        return list(sorted(results, key=operator.itemgetter(1), reverse=False))

    def find_items_for_user(self, user: str, positive: List[Tuple[str, EntityType]] = None,
                            negative: List[Tuple[str, EntityType]] = None, k=None) -> List[Tuple[str, float]]:
        assert self.fit_done
        if user not in self.users_set:
            raise UserNotFoundException("User with userID = %s, was not provided in training" % user)
        k = self.knn_params['n_neighbors'] if k is None else k
        embedding_list = [self.get_average_embeddings([(user, EntityType.USER)])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)
        (neighbors,), (dist,) = self.item_knn.knn_query([embedding], k=k)
        results = [(self.item_id_to_index.inverse[idx], dt) for idx, dt in zip(neighbors, dist)]
        return list(sorted(results, key=operator.itemgetter(1), reverse=False))

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
