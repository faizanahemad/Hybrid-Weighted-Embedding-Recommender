from typing import List, Dict, Tuple, Sequence, Type
from pandas import DataFrame


# TODO: Add Validations for add apis
# TODO: Use stable sort for item ranking
# TODO: Don't store the data just use it in embedding
# TODO: For Above provide cleanup/delete APIs
# TODO: Support Categorical features

class Feature:
    def __init__(self, feature_name: str, is_categorical: bool, cardinality: int):
        self.feature_name = feature_name
        self.is_categorical = is_categorical
        self.cardinality = cardinality


class RecommendationBase:
    def __init__(self):
        self.users = list()
        self.items = list()

        self.users_set = set(self.users)
        self.items_set = set(self.items)
        all_entities = self.users + self.items
        self.all_entity_set = set(all_entities)
        self.total_labels = len(self.all_entity_set)
        self.label_to_index = dict(zip(all_entities, list(range(self.total_labels))))
        self.index_to_label = dict(zip(list(range(self.total_labels)), all_entities))

    def add_users(self, users: List[str]):
        new_users_set = set(users)
        if len(new_users_set.intersection(self.all_entity_set)) > 0:
            raise AssertionError("Trying to Add Entity ID already present.")
        self.users.extend(users)
        self.all_entity_set.update(new_users_set)
        self.users_set.update(new_users_set)

        self.total_labels = len(self.all_entity_set)
        old_labels, old_indexes = zip(*self.label_to_index)
        self.label_to_index = dict(zip(old_labels + users, list(range(self.total_labels))))
        self.index_to_label = dict(zip(list(range(self.total_labels)), old_labels + users))

        return self

    def add_items(self, items: List[str]):
        new_items_set = set(items)
        if len(new_items_set.intersection(self.all_entity_set)) > 0:
            raise AssertionError("Trying to Add Entity ID already present.")
        self.items.extend(items)
        self.all_entity_set.update(new_items_set)
        self.items_set.update(new_items_set)

        self.total_labels = len(self.all_entity_set)
        old_labels, old_indexes = zip(*self.label_to_index)
        self.label_to_index = dict(zip(old_labels + items, list(range(self.total_labels))))
        self.index_to_label = dict(zip(list(range(self.total_labels)), old_labels + items))
        return self

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_data: Tuple[DataFrame, List[Feature]] = None,
            item_data: Tuple[DataFrame, List[Feature]] = None,
            user_item_affinities: Tuple[List[str], List[str], List[float]] = None,
            user_user_affinities: Tuple[List[str], List[str], List[float]] = None,
            item_item_affinities: Tuple[List[str], List[str], List[float]] = None,
            warm_start = True):
        # self.build_content_embeddings(item_data, user_item_affinities)
        assert user_data is None or \
               (len(user_data) == 2 and len(user_data[0].columns) == len(user_data[1]) and len(user_data[0]) == len(user_ids))
        assert item_data is None or \
               (len(item_data) == 2 and len(item_data[0].columns) == len(item_data[1]) and len(item_data[0]) == len(
                   item_ids))
        self.add_users(user_ids)
        self.add_items(item_ids)
        raise NotImplementedError()

    def default_predictions(self):
        pass

    def find_similar_items(self, item: str, positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def find_similar_users(self, user: str, positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def find_items_for_user(self, user: List[str], positive: List[str], negative: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
