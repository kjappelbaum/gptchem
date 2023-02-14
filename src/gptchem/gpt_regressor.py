from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner


class GPTRegressor:
    """Wrapper around GPT-3 fine tuning in style of a scikit-learn regressor."""

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        extractor: RegressionExtractor = RegressionExtractor(),
    ):
        """Initialize a GPTRegressor.

        Args:
            property_name (str): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            extractor (RegressionExtractor, optional): Callable object that can extract
                floats from the completions produced by the querier.
                Defaults to RegressionExtractor().
        """
        self.property_name = property_name
        self.tuner = tuner
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )
        self.extractor = extractor
        self.formatter = RegressionFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=property_name,
        )
        self.model_name = None
        self.tune_res = None

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Array of molecular representations.
            y (ArrayLike): Array of property values.
        """
        df = self._prepare_df(X, y)
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Array of molecular representations.

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        extracted = np.array(extracted, dtype=float)
        return extracted
