import abc
from typing import List, Union

import numpy as np

from ..logging import getLogger

Feature = List[Union[List[Union[str, List, int]], str]]


class BaseEmbed(metaclass=abc.ABCMeta):
    def __init__(self, n_dims, make_unit_length=True, **kwargs):
        self.n_dims = n_dims
        self.make_unit_length = make_unit_length
        self.kwargs = kwargs
        self.is_fit = False
        self.log = None

    @abc.abstractmethod
    def fit(self, feature: Feature, **kwargs):
        assert not self.is_fit and self.log is not None
        self.log.debug("Start Fitting...")
        self.is_fit = True

    @abc.abstractmethod
    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        assert self.is_fit
        pass

    def fit_transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.fit(feature, **kwargs)
        return self.check_output_dims(self.transform(feature, **kwargs), feature)

    def check_output_dims(self, output: np.ndarray, feature: Feature):
        if self.n_dims != output.shape[1] or output.shape[0] != len(feature):
            raise ValueError(
                "Unmatched Dims. Output Dims = %s, Required Dims = (%s,%s)" % (output.shape, len(feature), self.n_dims))
        return output


class IdentityEmbedding(BaseEmbed):
    def __init__(self, n_dims, **kwargs):
        super().__init__(n_dims, **kwargs)
        self.log = getLogger(type(self).__name__)
        self.columns = None

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        outputs = np.array(feature)
        return self.check_output_dims(outputs, feature)


class FeatureHashingEmbed(BaseEmbed):
    def __init__(self, n_dims=128, **kwargs):
        super().__init__(n_dims, make_unit_length=True)
        from sklearn.feature_extraction.text import HashingVectorizer
        self.log = getLogger(type(self).__name__)
        self.vectorizer = HashingVectorizer(n_features=n_dims, ngram_range=kwargs.pop("ngram_range", (1, 2)), analyzer=kwargs.pop("analyzer", "char_wb"))

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        outputs = self.vectorizer.fit_transform(feature).toarray()
        return self.check_output_dims(outputs, feature)
