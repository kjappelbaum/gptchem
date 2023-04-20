from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from gptchem.extractor import ClassificationExtractor, RegressionExtractor
from gptchem.formatter import ClassificationFormatter, RegressionFormatter
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


class BinnedGPTRegressor:
    """Wrapper around GPT-3 for "regression"
    by binning the property values in sufficiently many bins.

    The predicted property values are the bin centers.
    """

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        desired_accuracy: float = 0.1,
        equal_bin_sizes: bool = False,
        extractor: ClassificationExtractor = ClassificationExtractor(),
    ):
        """Initialize a BinnedGPTRegressor.

        Args:
            property_name (str): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            desired_accuracy (float, optional): Desired accuracy of the binning.
                Defaults to 0.1.
            equal_bin_sizes (bool, optional): Whether to use equal bin sizes.
                If False, the bin sizes are chosen such that the number of
                samples in each bin is approximately equal.
                Defaults to False.
            extractor (ClassificationExtractor, optional): Callable object that can extract
                floats from the completions produced by the querier.
                Defaults to ClassificationExtractor().
        """
        self.property_name = property_name
        self.tuner = tuner
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )
        self.extractor = extractor
        self.formatter = None
        self.model_name = None
        self.tune_res = None
        self.desired_accuracy = desired_accuracy
        self.equal_bin_sizes = equal_bin_sizes

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
        # set num bins
        num_bins = int(np.ceil((max(y) - min(y)) / self.desired_accuracy))

        # set formatter
        self.formatter = ClassificationFormatter(
            representation_column="repr",
            label_column="prop",
            property_name=self.property_name,
            num_classes=num_bins,
            qcut=not self.equal_bin_sizes,
        )

        df = self._prepare_df(X, y)
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def bin_indices_to_ranges(self, predicted_bin_indices: ArrayLike):
        """Convert a list of predicted bin indices to a list of bin ranges

        Use the bin edges from self.formatter.bins

        Args:
            predicted_bin_indices (ArrayLike): List of predicted bin indices

        Returns:
            ArrayLike: List of bin range tuples
        """
        bin_ranges = []
        for bin_index in predicted_bin_indices:
            bin_ranges.append((self.formatter.bins[bin_index], self.formatter.bins[bin_index + 1]))
        return bin_ranges

    def predict(self, X: ArrayLike, remap: bool = True) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Array of molecular representations.
            remap (bool, optional): Whether to remap the predicted bin indices to the

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        extracted = np.array(extracted, dtype=int)
        if remap:
            # self.formatter.bins is the list of bin edges
            # we want to remap the bin indices to the bin centers
            # so we take the average of each bin edge with the next one
            # for the first and last bin we just take the right or left edge, respectively
            centers = [self.formatter.bins[1]]
            for i in range(1, len(self.formatter.bins) - 2):
                centers.append((self.formatter.bins[i] + self.formatter.bins[i + 1]) / 2)
            centers.append(self.formatter.bins[-2])

            extracted = [centers[i] for i in extracted]
        return extracted
