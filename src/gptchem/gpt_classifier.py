from typing import Optional

import numpy as np
import pandas as pd
from fastcore.foundation import L
from numpy.typing import ArrayLike

from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner


class GPTClassifier:
    """Wrapper around GPT-3 fine tuning in style of a scikit-learn classifier."""

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
    ):
        self.property_name = property_name
        self.tuner = tuner
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )
        self.extractor = extractor
        self.formatter = ClassificationFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=property_name,
            num_classes=None,
        )
        self.model_name = None
        self.tune_res = None

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike):
        df = self._prepare_df(X, y)
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def predict(self, X: ArrayLike):
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        print(completions)
        extracted = self.extractor(completions)
        extracted = L([x if x is not None else np.nan for x in extracted])
        return extracted
