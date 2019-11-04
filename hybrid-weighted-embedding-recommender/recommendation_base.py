from typing import List, Dict, Tuple, Sequence, Type
from pandas import DataFrame


# TODO: Add Validations for add apis
# TODO: Use stable sort for item ranking
# TODO: Don't store the data just use it in embedding
# TODO: For Above provide cleanup/delete APIs
# TODO: Support Categorical features


class Feature:
    def __init__(self, feature_name: str, feature_type: str, values:List,
                 num_categories: int = 0):
        self.feature_name: str = feature_name
        self.feature_type: str = feature_type
        assert feature_type in ["id", "numeric", "text", "categorical", "multi_categorical"]
        if feature_type in ["categorical", "multi_categorical"] and num_categories == 0:
            raise ValueError("Specify Total Categories for Categorical Features")
        self.num_categories = num_categories
        self.values = values

    def __len__(self):
        return len(self.values)


class FeatureSet:
    def __init__(self, features: List[Feature]):
        self.features = features
        self.feature_names = [f.feature_name for f in features]
        self.feature_types = [f.feature_type for f in features]
        assert self.feature_types.count("text") <= 1
        assert self.feature_types.count("id") <= 1
        assert len(set([len(f) for f in features])) == 1

    def __len__(self):
        return len(self.feature_names)


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
            warm_start=True,
            **kwargs):
        # self.build_content_embeddings(item_data, user_item_affinities)
        self.add_users(user_ids)
        self.add_items(item_ids)
        raise NotImplementedError()

    def default_predictions(self):
        pass

    def find_similar_items(self, item: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        raise NotImplementedError()

    def find_similar_users(self, user: str, positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        raise NotImplementedError()

    def find_items_for_user(self, user: List[str], positive: List[str], negative: List[str]) -> List[Tuple[str, float]]:
        raise NotImplementedError()

    @staticmethod
    def persist(filename: str, instance):
        pass

    @staticmethod
    def load(filename: str):
        pass
