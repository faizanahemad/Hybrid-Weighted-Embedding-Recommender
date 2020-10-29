import numpy as np
import pandas as pd

from .BaseEmbed import BaseEmbed

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
from scipy.stats.mstats import rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_union

from ..logging import getLogger
from ..utils import unit_length
from typing import List, Union

Feature = List[List[Union[float, int]]]


class NumericEmbed(BaseEmbed):
    def __init__(self, n_dims, log=True, log1p=True, sqrt=True, quantile=True,
                 inverse=True, polynomial=False, power_transform=True,
                 cbrt=True, make_unit_length=True, n_iters=20, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.log_enabled = log
        self.log1p_enabled = log1p
        self.sqrt = sqrt
        self.cbrt = cbrt
        self.n_iters = n_iters
        self.sign = True
        self.scaler = None
        self.encoder = None
        self.inverse = inverse
        self.power_transform = power_transform
        self.quantile = quantile
        self.polynomial = polynomial
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else 0
        self.log = getLogger(type(self).__name__)

    def __prepare_inputs__(self, inputs):
        assert np.sum(np.isnan(inputs)) == 0
        assert np.sum(np.isinf(inputs)) == 0

        self.log_enabled = self.log_enabled and np.sum(inputs <= 1e-9) == 0
        self.sqrt = self.sqrt and np.sum(inputs < 0) == 0
        self.log1p_enabled = self.log1p_enabled and np.sum(inputs <= -1.0) == 0
        self.sign = self.sign and not self.log_enabled
        results = inputs.copy()
        if self.sign:
            results = np.concatenate((results, np.sign(inputs)), axis=1)
        if self.log_enabled:
            results = np.concatenate((results, np.log(inputs)), axis=1)
        if self.log1p_enabled:
            results = np.concatenate((results, np.log1p(inputs)), axis=1)
        if self.sqrt:
            results = np.concatenate((results, np.sqrt(inputs)), axis=1)
        if self.cbrt:
            results = np.concatenate((results, np.cbrt(inputs)), axis=1)
        if self.inverse:
            results = np.concatenate((results, 1 / (inputs + 1e-3)), axis=1)

        if self.polynomial and isinstance(self.polynomial, bool):
            self.polynomial = PolynomialFeatures(interaction_only=True, include_bias=False)
            results = np.concatenate((results, self.polynomial.fit_transform(np.concatenate((results, 1 / (inputs + 1e-3)), axis=1))), axis=1)
        elif self.polynomial:
            results = np.concatenate((results, self.polynomial.transform(np.concatenate((results, 1 / (inputs + 1e-3)), axis=1))), axis=1)

        if self.power_transform and isinstance(self.power_transform, bool):
            self.power_transform = PowerTransformer()
            results = np.concatenate((results, self.power_transform.fit_transform(inputs)), axis=1)
        elif self.power_transform:
            results = np.concatenate((results, self.power_transform.transform(inputs)), axis=1)

        if self.quantile and isinstance(self.quantile, bool):
            self.quantile = QuantileTransformer(n_quantiles=100, random_state=0)
            results = np.concatenate((results, self.quantile.fit_transform(inputs)), axis=1)
        elif self.quantile:
            results = np.concatenate((results, self.quantile.transform(inputs)), axis=1)

        results = np.concatenate((results, np.square(inputs)), axis=1)

        if self.scaler is None:
            self.scaler = make_union(MinMaxScaler(feature_range=(-0.95, 0.95)), StandardScaler())
            results = self.scaler.fit_transform(results)
        else:
            results = self.scaler.transform(results)
        return results

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        inputs = np.array(feature)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)

        pre_shape = inputs.shape
        inputs = self.__prepare_inputs__(inputs)
        self.log.info("PreShape PostShape = %s, %s" % (pre_shape, inputs.shape))
        encoder = IncrementalPCA(n_components=self.n_dims, whiten=True, batch_size=2**16)
        encoder.fit(inputs)
        self.log.info("Explained Variance Ratio = %s" % (sum(encoder.explained_variance_ratio_)))
        self.encoder = encoder
        self.log.debug("End Fitting NumericEmbed")

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.log.debug("Start Transform NumericEmbed")
        inputs = np.array(feature)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        inputs = self.__prepare_inputs__(inputs)
        outputs = self.encoder.transform(inputs)
        assert np.sum(np.isnan(outputs)) == 0
        assert np.sum(np.isinf(outputs)) == 0
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        self.log.debug("End Transform NumericEmbedd")
        return self.check_output_dims(outputs, feature)
