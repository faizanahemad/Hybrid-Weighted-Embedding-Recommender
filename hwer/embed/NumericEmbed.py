
from .BaseEmbed import BaseEmbed
import abc
import os

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
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
Feature = List[List[Union[float, int]]]

class NumericEmbed(BaseEmbed):
    def __init__(self, n_dims, log=True, log1p=True, sqrt=True,
                 make_unit_length=True, n_iters=20, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.log_enabled = log
        self.log1p_enabled = log1p
        self.sqrt = sqrt
        self.n_iters = n_iters
        self.sign = True
        self.scaler = None
        self.encoder = None
        self.standard_scaler = None
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else 0
        self.log = getLogger(type(self).__name__)

    def __prepare_inputs__(self, inputs):
        assert np.sum(np.isnan(inputs)) == 0
        assert np.sum(np.isinf(inputs)) == 0
        if self.standard_scaler is None:
            self.standard_scaler = StandardScaler()
            standardized_inputs = self.standard_scaler.fit_transform(inputs)
        else:
            standardized_inputs = self.standard_scaler.transform(inputs)

        self.log_enabled = self.log_enabled and np.sum(inputs <= 1e-9) == 0
        self.sqrt = self.sqrt and np.sum(inputs < 0) == 0
        self.log1p_enabled = self.log1p_enabled and np.sum(inputs <= -1.0) == 0
        self.sign = self.sign and not self.log_enabled
        results = np.concatenate((inputs, standardized_inputs), axis=1)
        if self.sign:
            results = np.concatenate((results, np.sign(inputs)), axis=1)
        if self.log_enabled:
            results = np.concatenate((results, np.log(inputs)), axis=1)
        if self.log1p_enabled:
            results = np.concatenate((results, np.log1p(inputs)), axis=1)
        if self.sqrt:
            results = np.concatenate((results, np.sqrt(inputs)), axis=1)
        results = np.concatenate((results, np.square(inputs)), axis=1)

        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(-0.95, 0.95))
            results = self.scaler.fit_transform(results)
        else:
            results = self.scaler.transform(results)
        return results

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        inputs = np.array(feature)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)

        ranks = rankdata(inputs, axis=0) / len(inputs)

        inputs = self.__prepare_inputs__(inputs)
        mean, var = self.standard_scaler.mean_, self.standard_scaler.var_
        mean = np.broadcast_to(mean, (len(inputs),len(mean)))
        var = np.broadcast_to(var, (len(inputs), len(var)))
        outputs = np.concatenate((inputs, ranks, mean, var), axis=1)

        _, encoder = auto_encoder_transform(inputs, outputs, n_dims=self.n_dims, verbose=self.verbose,
                                            epochs=self.n_iters)
        self.encoder = encoder
        self.log.debug("End Fitting NumericEmbed")

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.log.debug("Start Transform NumericEmbed")
        inputs = np.array(feature)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        inputs = self.__prepare_inputs__(inputs)
        outputs = self.encoder.predict(inputs)
        assert np.sum(np.isnan(outputs)) == 0
        assert np.sum(np.isinf(outputs)) == 0
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        self.log.debug("End Transform NumericEmbedd")
        return self.check_output_dims(outputs, feature)
