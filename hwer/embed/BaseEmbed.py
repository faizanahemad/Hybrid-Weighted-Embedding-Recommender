import abc
import os

import numpy as np
import pandas as pd
from scipy.stats.mstats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..logging import getLogger
from ..utils import auto_encoder_transform, unit_length, clean_text, is_1d_array
from ..utils import is_num, is_2d_array, NodeNotFoundException
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set, Union

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
            raise ValueError("Unmatched Dims. Output Dims = %s, Required Dims = (%s,%s)" % (output.shape, len(feature), self.n_dims))
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
