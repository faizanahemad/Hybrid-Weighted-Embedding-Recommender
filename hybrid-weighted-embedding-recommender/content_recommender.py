from .recommendation_base import RecommendationBase, Feature
from typing import List, Dict, Tuple, Sequence, Type
from pandas import DataFrame

class ContentRecommendation(RecommendationBase):
    def __init__(self, ndims=32):
        super().__init__()
        self.ndims = 32

    def add_users(self, users: List[str]):
        super().add_users(users)

    def add_items(self, items: List[str]):
        super().add_items(items)

    def fit(self,
            user_ids: List[str],
            item_ids: List[str],
            user_data: Tuple[DataFrame, List[Feature]],
            item_data: Tuple[DataFrame, List[Feature]],
            user_item_affinities: Tuple[List[str], List[str], List[float]] = None,
            user_user_affinities: Tuple[List[str], List[str], List[float]] = None,
            item_item_affinities: Tuple[List[str], List[str], List[float]] = None,
            warm_start=True):
        pass

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
