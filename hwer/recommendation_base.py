from typing import List, Dict, Tuple, Sequence, Type
from pandas import DataFrame
import abc
import numpy as np
import nmslib


# TODO: Add Validations for add apis
# TODO: Use stable sort for item ranking
# TODO: Don't store the data just use it in embedding
# TODO: For Above provide cleanup/delete APIs
# TODO: Support Categorical features


class Feature:
    def __init__(self, feature_name: str,
                 feature_type: str,
                 feature_dtype: type,
                 values: List,
                 num_categories: int = 0):
        """

        :param feature_name:
        :param feature_type: Supported Types ["id", "numeric", "str", "categorical", "multi_categorical"]
        :param feature_dtype:
        :param values:
        :param num_categories:
        """
        self.feature_name: str = feature_name
        self.feature_type: str = feature_type
        assert feature_type in ["id", "numeric", "str", "categorical", "multi_categorical"]
        if feature_type in ["categorical", "multi_categorical"] and num_categories == 0:
            raise ValueError("Specify Total Categories for Categorical Features")

        self.num_categories = num_categories
        self.values = values
        self.feature_dtype = feature_dtype

    def __len__(self):
        return len(self.values)


class FeatureSet:
    def __init__(self, features: List[Feature]):
        self.features = features
        self.feature_names = [f.feature_name for f in features]
        self.feature_types = [f.feature_type for f in features]
        self.feature_dtypes = [f.feature_dtype for f in features]
        assert self.feature_types.count("text") <= 1
        assert self.feature_types.count("id") <= 1
        # check all features have same count of values in FeatureSet
        assert len(set([len(f) for f in features])) == 1

    def __getitem__(self, key):
        return self.features[key]


# TODO: Use hnswlib for storing and retrieving user and item vectors
class RecommendationBase(metaclass=abc.ABCMeta):
    def __init__(self, knn_params: dict, n_output_dims: int = 32,):
        self.users = list()
        self.items = list()
        self.users_set = set(self.users)
        self.items_set = set(self.items)

        self.user_features = None
        self.item_features = None
        self.item_only_features = None
        self.user_only_features = None

        self.user_id_to_index = None
        self.index_to_user_id = None
        self.item_id_to_index = None
        self.index_to_item_id = None
        self.user_id_to_vector = None
        self.item_id_to_vector = None
        self.user_knn = None
        self.item_knn = None
        self.fit_done = False
        self.knn_params = knn_params
        if self.knn_params is None:
            self.knn_params = dict(n_neighbors=1000,
                                index_time_params = {'M': 15, 'indexThreadQty': 16, 'efConstruction': 200, 'post': 0, 'delaunay_type': 1})

        self.n_output_dims = n_output_dims

    def __add_users__(self, users: List[str]):
        new_users_set = set(users)
        if len(new_users_set.intersection(self.users_set)) > 0:
            raise AssertionError("Trying to Add User ID already present.")
        self.users.extend(users)
        self.users_set.update(new_users_set)

        self.user_id_to_index = dict(zip(self.users, list(range(len(self.users)))))
        self.index_to_user_id = dict(zip(list(range(len(self.users))), self.users))
        return self

    def __add_items__(self, items: List[str]):
        new_items_set = set(items)
        if len(new_items_set.intersection(self.items_set)) > 0:
            raise AssertionError("Trying to Add Item ID already present.")
        self.items.extend(items)
        self.items_set.update(new_items_set)

        self.item_id_to_index = dict(zip(self.items, list(range(len(self.items)))))
        self.index_to_item_id = dict(zip(list(range(len(self.items))), self.items))
        return self

    def __build_knn__(self, user_ids: List[str], item_ids: List[str],
                      user_vectors: np.ndarray, item_vectors: np.ndarray):
        self.user_id_to_vector = dict(zip(user_ids, user_vectors))
        self.item_id_to_vector = dict(zip(item_ids, user_vectors))
        n_neighbors = self.knn_params["n_neighbors"]
        index_time_params = self.knn_params["index_time_params"]
        query_time_params = {'efSearch': n_neighbors}

        nms_user_index = nmslib.init(method='hnsw', space='cosinesimil')
        nms_user_index.addDataPointBatch(user_vectors)
        nms_user_index.createIndex(index_time_params, print_progress=True)
        nms_user_index.setQueryTimeParams(query_time_params)

        nms_item_index = nmslib.init(method='hnsw', space='cosinesimil')
        nms_item_index.addDataPointBatch(item_vectors)
        nms_item_index.createIndex(index_time_params, print_progress=True)
        nms_item_index.setQueryTimeParams(query_time_params)
        self.user_knn = nms_user_index
        self.item_knn = nms_item_index
        return nms_user_index, nms_item_index

    @abc.abstractmethod
    def add_user(self, user_id: str, features: FeatureSet, user_item_affinities: List[Tuple[str, str, float]]):
        assert self.fit_done
        self.__add_users__(users=[user_id])
        assert set([f.feature_name for f in features if f.feature_type != "id"]) == set(self.user_features)
        users, items, weights = zip(*user_item_affinities)
        embedding = np.average(self.get_average_embeddings(items), weights=weights)
        self.user_id_to_vector[user_id] = embedding




    @abc.abstractmethod
    def add_item(self, item_id, features: FeatureSet, user_item_affinities: List[Tuple[str, str, float]]):
        assert self.fit_done
        self.__add_items__(items=[item_id])
        assert set([f.feature_name for f in features if f.feature_type != "id"]) == set(self.item_features)
        users, items, weights = zip(*user_item_affinities)
        embedding = np.average(self.get_average_embeddings(users), weights=weights)
        self.item_id_to_vector[item_id] = embedding

    @abc.abstractmethod
    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            **kwargs):
        # self.build_content_embeddings(item_data, user_item_affinities)
        assert not self.fit_done
        assert len(user_ids) == len(set(user_ids))
        assert len(item_ids) == len(set(item_ids))
        self.__add_users__(user_ids)
        self.__add_items__(item_ids)

        item_data: FeatureSet = kwargs["item_data"]
        user_data: FeatureSet = kwargs["user_data"]
        self.user_features = [feature.feature_name for feature in user_data if feature.feature_type != "id"]
        self.item_features = [feature.feature_name for feature in item_data if feature.feature_type != "id"]
        self.item_only_features = list(set(self.item_features) - set(self.user_features))
        self.user_only_features = list(set(self.user_features) - set(self.item_features))

        return

    @abc.abstractmethod
    def default_predictions(self):
        return

    def get_average_embeddings(self, entities: List[str]):
        embeddings = []
        for entity in entities:
            if entity in self.user_id_to_vector:
                embeddings.append(self.user_id_to_vector[entity])
            elif entity in self.item_id_to_vector:
                embeddings.append(self.item_id_to_vector[entity])
            else:
                raise ValueError("Unseen entity: %s" % (entity))
        return np.average(embeddings, axis=0)

    def find_similar_items(self, item: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert item in self.item_id_to_vector
        embedding_list = [self.item_id_to_vector[item], self.get_average_embeddings(positive),
                          -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.item_knn.knnQuery(embedding)
        return [(self.index_to_item_id[idx], dt) for idx, dt in zip(neighbors, dist)]

    def find_similar_users(self, user: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert user in self.user_id_to_vector
        embedding_list = [self.user_id_to_vector[user], self.get_average_embeddings(positive),
                          -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.user_knn.knnQuery(embedding)
        return [(self.index_to_user_id[idx], dt) for idx, dt in zip(neighbors, dist)]

    def find_items_for_user(self, user: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        assert self.fit_done
        assert user in self.user_id_to_vector
        embedding_list = [self.user_id_to_vector[user], self.get_average_embeddings(positive),
                          -1 * self.get_average_embeddings(negative)]
        embedding = np.average(embedding_list, axis=0)
        neighbors, dist = self.item_knn.knnQuery(embedding)
        return [(self.index_to_item_id[idx], dt) for idx, dt in zip(neighbors, dist)]

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
