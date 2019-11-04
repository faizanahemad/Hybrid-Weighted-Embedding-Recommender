from .recommendation_base import Feature
import numpy as np


class ContentEmbedderBase:
    def __init__(self, n_dims, make_unit_length=True, **kwargs):
        self.n_dims = n_dims
        self.make_unit_length = make_unit_length
        self.kwargs = kwargs

    def fit(self, feature: Feature, **kwargs):
        raise NotImplementedError()

    def predict(self, feature: Feature, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def fit_predict(self, feature: Feature, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class CategoricalEmbedder(ContentEmbedderBase):
    # Take Categorical Encoder from data-science-utils
    pass


class FasttextEmbedder(ContentEmbedderBase):
    pass


class NumericEmbedder(ContentEmbedderBase):
    # Take Each Numerical Feature -> Log, sqrt, square, -> Normalizer -> AutoEncoder
    pass


class FlairGlove100Embedder(ContentEmbedderBase):
    pass


class FlairGlove100AndBytePairEmbedder(ContentEmbedderBase):
    pass