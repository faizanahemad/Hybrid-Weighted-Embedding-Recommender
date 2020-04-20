
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
Feature = List[Union[List[str], str]]


class FlairGlove100Embed(BaseEmbed):
    def __init__(self, n_dims=100, make_unit_length=True):
        from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence, BytePairEmbeddings
        super().__init__(n_dims=n_dims, make_unit_length=make_unit_length)
        embeddings = [WordEmbeddings('glove')]
        self.embeddings = DocumentPoolEmbeddings(embeddings, fine_tune_mode='none')
        self.log = getLogger(type(self).__name__)

    def get_sentence_vector(self, text):
        from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence, BytePairEmbeddings
        sentence = Sentence(clean_text(text))
        _ = self.embeddings.embed(sentence)
        a = sentence.get_embedding()
        result = a.detach().cpu().numpy()
        if np.sum(result[0:5]) == 0:
            result = np.random.randn(self.n_dims)
        return result

    def fit(self, feature: Feature, **kwargs):
        super().fit(feature, **kwargs)
        self.log.debug("End Fitting FlairEmbedding")
        return

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.log.debug("Start Transform TextEmbedding...")
        outputs = np.vstack([np.array([self.get_sentence_vector(i) for i in t]).mean(0) if is_1d_array(t) else self.get_sentence_vector(t) for t in tqdm(feature)])
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        self.log.debug("End Transform TextEmbedding")
        return self.check_output_dims(outputs, feature)


class FlairGlove100AndBytePairEmbed(FlairGlove100Embed):
    def __init__(self, n_dims=200, make_unit_length=True):
        from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence, BytePairEmbeddings, StackedEmbeddings
        super().__init__(n_dims=n_dims, make_unit_length=make_unit_length)
        embeddings = [WordEmbeddings('glove'), BytePairEmbeddings('en')]
        self.embeddings = DocumentPoolEmbeddings(embeddings, fine_tune_mode='none')
        self.log = getLogger(type(self).__name__)

