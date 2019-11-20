from .recommendation_base import Feature, FeatureSet, FeatureType, EntityType
import numpy as np
import abc
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from .utils import auto_encoder_transform, unit_length, clean_text
import fasttext
import os
from scipy.stats.mstats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, BytePairEmbeddings, StackedEmbeddings
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Type, Union
from sklearn.preprocessing import StandardScaler


class ContentEmbeddingBase(metaclass=abc.ABCMeta):
    def __init__(self, n_dims, make_unit_length=True, **kwargs):
        self.n_dims = n_dims
        self.make_unit_length = make_unit_length
        self.kwargs = kwargs
        self.is_fit = False

    @abc.abstractmethod
    def fit(self, feature: Union[Feature, FeatureSet], **kwargs):
        assert not self.is_fit
        self.is_fit = True

    @abc.abstractmethod
    def transform(self, feature: Union[Feature, FeatureSet], **kwargs) -> np.ndarray:
        assert self.is_fit

    def fit_transform(self, feature: Union[Feature, FeatureSet], **kwargs) -> np.ndarray:
        self.fit(feature, **kwargs)
        if type(feature) == FeatureSet:
            output_check_feature = feature.features[0]
        else:
            output_check_feature = feature
        return self.check_output_dims(self.transform(feature, **kwargs), output_check_feature)

    def check_output_dims(self, output: np.ndarray, feature: Feature):
        if self.n_dims != output.shape[1] or output.shape[0] != len(feature):
            raise ValueError("Unmatched Dims. Output Dims = %s, Required Dims = (%s,%s)"% (output.shape, len(feature), self.n_dims))
        return output


class CategoricalEmbedding(ContentEmbeddingBase):
    def __init__(self, n_dims, make_unit_length=True, n_iters=50, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.n_iters = n_iters
        self.encoder = None
        self.ohe = None

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert type(feature.values[0]) == str
        df = pd.DataFrame(data=feature.values, columns=["input"], dtype=str)
        # pd.DataFrame(data=np.array([feature.values]).reshape(-1,1))

        if "target" in kwargs:
            target: FeatureSet = kwargs["target"]
            assert set([len(f) for f in target.features]) == {len(feature)}

            for i, feat in enumerate(target.features):
                df['target_'+str(i)] = feat.values
                # mode -> pd.Series.mode
            df = df.groupby(["input"]).agg(['mean', 'median', 'min', 'max', 'std', 'count']).reset_index().fillna(0)
        else:
            df = df.groupby(["input"], as_index=False).agg(lambda x: set(x))

        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        network_inputs = ohe.fit_transform(df[['input']])
        network_output = np.concatenate((network_inputs, df.drop(columns=["input"])), axis=1)

        min_max_scaler = MinMaxScaler(feature_range=(-0.95, 0.95))
        network_output = min_max_scaler.fit_transform(network_output)

        _, encoder = auto_encoder_transform(network_inputs, network_output, n_dims=self.n_dims, verbose=0, epochs=self.n_iters)
        self.encoder = encoder
        self.ohe = ohe

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        df = pd.DataFrame(data=feature.values, columns=["input"], dtype=str)
        network_inputs = self.ohe.transform(df[['input']])

        outputs = unit_length(self.encoder.predict(network_inputs), axis=1) if self.make_unit_length else self.encoder.predict(network_inputs)
        return self.check_output_dims(outputs, feature)


class MultiCategoricalEmbedding(ContentEmbeddingBase):
    def __init__(self, n_dims, make_unit_length=True, n_iters=50, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.n_iters = n_iters
        self.encoder = None
        self.vectorizer = None
        self.input_mapper = lambda x: " ".join(map(lambda y: "__" + str(y).strip() + "__", x))

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert type(feature.values[0]) == list or type(feature.values[0]) == np.ndarray
        assert feature.feature_type == FeatureType.MULTI_CATEGORICAL
        df = pd.DataFrame(data=np.array(feature.values).T, columns=["input"])

        if "target" in kwargs:
            target: FeatureSet = kwargs["target"]
            assert set(target.feature_types) == {FeatureType.NUMERIC}
            assert set([len(f) for f in target.features]) == {len(feature)}

            for i, feat in enumerate(target.features):
                df['target_'+str(i)] = feat.values
                # https://stackoverflow.com/questions/49434712/pandas-groupby-on-a-column-of-lists
            df = df.groupby(df['input'].map(tuple)).agg(['mean', 'median', 'min', 'max', 'std', 'count']).reset_index().fillna(0)
        else:
            df = df.groupby(df['input'].map(tuple)).agg(['count']).reset_index()
            df.columns = ['input', 'count']

        vectorizer = CountVectorizer()
        network_inputs = vectorizer.fit_transform(list(df.input.map(self.input_mapper).values)).toarray()
        network_output = np.concatenate((network_inputs, df.drop(columns=["input"])), axis=1)

        min_max_scaler = MinMaxScaler(feature_range=(-0.95, 0.95))
        network_output = min_max_scaler.fit_transform(network_output)

        _, encoder = auto_encoder_transform(network_inputs, network_output, n_dims=self.n_dims, verbose=0, epochs=self.n_iters)
        self.encoder = encoder
        self.vectorizer = vectorizer

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        df = pd.DataFrame(data=np.array(feature.values).T, columns=["input"])
        network_inputs = self.vectorizer.transform(list(df.input.map(self.input_mapper).values)).toarray()
        outputs = unit_length(self.encoder.predict(network_inputs), axis=1) if self.make_unit_length else self.encoder.predict(network_inputs)
        return self.check_output_dims(outputs, feature)


class FasttextEmbedding(ContentEmbeddingBase):
    def __init__(self, n_dims, fasttext_file=None, make_unit_length=True, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.fasttext_file = fasttext_file
        self.text_model = None
        self.fasttext_params = kwargs["fasttext_params"] \
            if "fasttext_params" in kwargs else dict(neg=5, ws=7, minCount=3, bucket=1000000, minn=4, maxn=5,
                                                     dim=self.n_dims, epoch=10, lr=0.1, thread=os.cpu_count())

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert feature.feature_type == FeatureType.STR
        if self.fasttext_file is None:
            df = pd.DataFrame(data=feature.values, columns=[feature.feature_name])
            df[feature.feature_name] = df[feature.feature_name].fillna("").apply(clean_text)
            df.to_csv("sentences.txt",index=False)
            text_model = fasttext.train_unsupervised("sentences.txt", "skipgram", **self.fasttext_params)
        else:
            text_model = fasttext.load_model(self.fasttext_file)
        self.text_model = text_model

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        assert feature.feature_type == FeatureType.STR
        outputs = list(map(lambda x: self.text_model.get_sentence_vector(clean_text(x)), feature.values))
        outputs = np.array(outputs)
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        return self.check_output_dims(outputs, feature)


class NumericEmbedding(ContentEmbeddingBase):
    def __init__(self, n_dims, log=True, sqrt=True, square=True, percentile=True,
                 make_unit_length=True, n_iters=50, **kwargs):
        super().__init__(n_dims, make_unit_length, **kwargs)
        self.log = log
        self.sqrt = sqrt
        self.square = square
        self.percentile = percentile
        self.n_iters = n_iters
        self.scaler = None
        self.encoder = None
        self.standard_scaler = None

    def __prepare_inputs(self, inputs):
        scaled_inputs = self.scaler.transform(inputs)
        standardized_inputs = self.standard_scaler.transform(inputs)
        outputs = np.concatenate((scaled_inputs, standardized_inputs), axis=1)
        if self.log and np.sum(inputs <= 0) == 0:
            outputs = np.concatenate((outputs, np.log(inputs)), axis=1)
        if self.sqrt and np.sum(inputs <= 0) == 0:
            outputs = np.concatenate((outputs, np.sqrt(inputs)), axis=1)
        if self.square:
            outputs = np.concatenate((outputs, np.square(inputs)), axis=1)
        if self.percentile:
            outputs = np.concatenate((outputs, rankdata(inputs, axis=0) / len(inputs)), axis=1)

        inputs = outputs.copy()
        return inputs

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert feature.feature_type == FeatureType.NUMERIC
        inputs = np.array(feature.values)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-0.95, 0.95))
        standard_scaler = StandardScaler()
        _ = scaler.fit(inputs)
        _ = standard_scaler.fit(inputs)
        self.scaler = scaler
        self.standard_scaler = standard_scaler

        inputs = self.__prepare_inputs(inputs)

        _, encoder = auto_encoder_transform(inputs, inputs.copy(), n_dims=self.n_dims, verbose=0,
                                            epochs=self.n_iters)
        self.encoder = encoder

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        assert feature.feature_type == FeatureType.NUMERIC
        inputs = np.array(feature.values)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        inputs = self.__prepare_inputs(inputs)
        outputs = unit_length(self.encoder.predict(inputs),
                              axis=1) if self.make_unit_length else self.encoder.predict(inputs)
        return self.check_output_dims(outputs, feature)


class FlairGlove100Embedding(ContentEmbeddingBase):
    def __init__(self, make_unit_length=True):
        super().__init__(n_dims=100, make_unit_length=make_unit_length)
        embeddings = [WordEmbeddings('glove')]
        self.embeddings = DocumentPoolEmbeddings(embeddings)

    def get_sentence_vector(self, text):
        sentence = Sentence(clean_text(text))
        _ = self.embeddings.embed(sentence)
        a = sentence.get_embedding()
        result = a.detach().cpu().numpy()
        if np.sum(result[0:5]) == 0:
            result = np.random.randn(self.n_dims)
        return result

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert feature.feature_type == FeatureType.STR
        return

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        assert feature.feature_type == FeatureType.STR
        outputs = np.vstack([self.get_sentence_vector(t) for t in tqdm(feature.values)])
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        return self.check_output_dims(outputs, feature)


class FlairGlove100AndBytePairEmbedding(ContentEmbeddingBase):
    def __init__(self, make_unit_length=True):
        super().__init__(n_dims=200, make_unit_length=make_unit_length)
        embeddings = [WordEmbeddings('glove'), BytePairEmbeddings('en')]
        self.embeddings = DocumentPoolEmbeddings(embeddings)

    def get_sentence_vector(self, text):
        sentence = Sentence(clean_text(text))
        _ = self.embeddings.embed(sentence)
        a = sentence.get_embedding()
        result = a.detach().numpy()
        if np.sum(result[0:5]) == 0:
            result = np.random.randn(self.n_dims)
        return result

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        assert feature.feature_type == FeatureType.STR
        return

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        assert feature.feature_type == FeatureType.STR
        outputs = np.vstack([self.get_sentence_vector(t) for t in tqdm(feature.values)])
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        return self.check_output_dims(outputs, feature)
