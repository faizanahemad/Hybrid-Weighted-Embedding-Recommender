import numpy as np
import pandas as pd

from .BaseEmbed import BaseEmbed

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from ..logging import getLogger
from ..utils import unit_length, is_1d_array
from typing import List, Union

Feature = List[List[Union[str, List, int]]]


class CategoricalEmbed(BaseEmbed):
    def __init__(self, n_dims, make_unit_length=True, n_iters=20, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.n_iters = n_iters
        self.encoder = None
        self.ohe = None
        self.vectorizers = None
        self.log = getLogger(type(self).__name__)
        self.columns = None
        self.input_mapper = lambda x: " ".join(map(lambda y: "__" + str(y).strip() + "__", x))
        self.verbose = kwargs["verbose"] if "verbose" in kwargs else 0

    def __build_input__(self, df):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False) if self.ohe is None else self.ohe
        self.ohe = ohe
        network_inputs = ohe.fit_transform(df[self.categorical_columns])
        vectorizers = dict() if self.vectorizers is None else self.vectorizers
        self.vectorizers = vectorizers
        for c in self.multi_columns:
            vectorizer = CountVectorizer() if c not in vectorizers else vectorizers[c]
            self.vectorizers[c] = vectorizer
            inputs = vectorizer.fit_transform(list(df[c].map(self.input_mapper).values)).toarray()
            network_inputs = np.concatenate((network_inputs, inputs), axis=1)
        return network_inputs

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert is_1d_array(feature[0])
        columns = list(range(len(feature[0])))
        self.columns = columns
        df = pd.DataFrame(data=feature, columns=columns, dtype=str)
        # df = df.groupby(columns, as_index=False).agg(set)
        inputs = df[columns]

        df['sentinel'] = 1
        # Separate categorical and multi-categorical
        categorical_columns = []
        multi_columns = []
        columns_count = []
        for column in columns:
            if type(df[column].values[0]) == str or type(df[column].values[0]) == int:
                categorical_columns.append(column)
            elif type(df[column].values[0]) == list or type(df[column].values[0]) == tuple or type(
                    df[column].values[0]) == np.ndarray:
                multi_columns.append(column)
                df[column] = df[column].map(tuple)
            else:
                raise ValueError("CategoricalEmbed: Failed to classify column")
            cc = df.groupby([column])[["sentinel"]].agg(['count']).reset_index().fillna(0)
            cn = str(column) + "_count"
            cc.columns = [column, cn]
            df = df.merge(cc, on=column)
            columns_count.append(cn)
        self.categorical_columns = categorical_columns
        self.multi_columns = multi_columns

        df["join"] = df[columns].apply(tuple, axis=1)
        global_counts = df.groupby("join", as_index=False)[["sentinel"]].agg(['count']).reset_index().fillna(0)
        global_counts.columns = ["join", "count"]
        df = df.merge(global_counts[["join", "count"]], on="join")
        df = df.drop(columns=["sentinel", "join"])
        df[columns] = inputs
        columns_count.append("count")
        # df = df.groupby(pd.Series(list(map(tuple, df[columns].values))), as_index=False).agg(['count']).reset_index()
        network_inputs = self.__build_input__(df)
        network_output = np.concatenate((network_inputs, df[columns_count].values), axis=1)
        min_max_scaler = MinMaxScaler(feature_range=(0.0, 0.95))
        network_output = min_max_scaler.fit_transform(network_output)
        from ..utils import auto_encoder_transform
        _, encoder = auto_encoder_transform(network_inputs, network_output, n_dims=self.n_dims, verbose=self.verbose,
                                            epochs=self.n_iters)
        self.encoder = encoder

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.log.debug("Start Transform CategoricalEmbedding...")
        df = pd.DataFrame(data=feature, columns=self.columns, dtype=str)
        network_inputs = self.__build_input__(df)
        outputs = self.encoder.predict(network_inputs)
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        self.log.debug("End Transform CategoricalEmbedding")
        return self.check_output_dims(outputs, feature)
