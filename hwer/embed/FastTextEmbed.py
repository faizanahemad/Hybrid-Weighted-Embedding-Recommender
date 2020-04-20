import os

import numpy as np
import pandas as pd

from .BaseEmbed import BaseEmbed

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
from tqdm import tqdm

from ..logging import getLogger
from ..utils import unit_length, clean_text, is_1d_array
from typing import List, Union

Feature = List[Union[List[str], str]]


class FastTextEmbed(BaseEmbed):
    def __init__(self, n_dims, fasttext_file=None, make_unit_length=True, **kwargs):
        super().__init__(n_dims, make_unit_length)
        self.fasttext_file = fasttext_file
        self.text_model = None
        self.fasttext_params = kwargs["fasttext_params"] \
            if "fasttext_params" in kwargs else dict(neg=10, ws=6, minCount=3, bucket=1000000, minn=4, maxn=5,
                                                     dim=self.n_dims, epoch=10, lr=0.05, thread=os.cpu_count())
        self.log = getLogger(type(self).__name__)

    def get_sentence_vector(self, text):
        result = self.text_model.get_sentence_vector(text)
        if np.sum(result[0:5]) == 0:
            result = np.random.randn(self.n_dims)
        return result

    def fit(self, feature: Feature, **kwargs):
        import fasttext
        super().fit(feature, **kwargs)
        if self.fasttext_file is None:
            sentences = [". ".join(f) if is_1d_array(f) else f for f in feature]
            df = pd.DataFrame(data=sentences, columns=["sentences"])
            df["sentences"] = df["sentences"].fillna("").apply(clean_text)
            sfile = "sentences-%s.txt" % str(np.random.randint(int(1e8)))
            df.to_csv(sfile, index=False)
            text_model = fasttext.train_unsupervised(sfile, "skipgram", **self.fasttext_params)
            import os
            try:
                os.remove(sfile)
            except FileNotFoundError:
                pass

        else:
            text_model = fasttext.load_model(self.fasttext_file)
        self.text_model = text_model
        self.log.debug("End Fitting FasttextEmbed")

    def transform(self, feature: Feature, **kwargs) -> np.ndarray:
        self.log.debug("Start Transform TextEmbedding...")
        outputs = np.vstack([np.array([self.get_sentence_vector(i) for i in t]).mean(0) if is_1d_array(
            t) else self.get_sentence_vector(t) for t in tqdm(feature)])
        outputs = unit_length(outputs, axis=1) if self.make_unit_length else outputs
        self.log.debug("End Transform TextEmbedding")
        return self.check_output_dims(outputs, feature)
