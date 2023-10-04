from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
        save_valid_file: bool = False,
    ):
        """Initialize a GPTClassifier.

        Args:
            property_name (str): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            extractor (ClassificationExtractor, optional): Callable object that can extract
                integers from the completions produced by the querier.
                Defaults to ClassificationExtractor().
            save_valid_file (bool, optional): Whether to save the validation file.
                Defaults to False.
        """
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
        self.save_valid_file = save_valid_file

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
        """
        df = self._prepare_df(X, y)
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))
        formatted = self.formatter(df)
        if self.save_valid_file:
            self.tuner._write_file(formatted, "valid")

        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        return extracted


class NGramGPTClassifier:
    """Add the predictions of a N-Gram model to the prompt.
    Empirically, this tends to degrade performance.
    """

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        count_vectorizer: Optional[CountVectorizer] = None,
        ngram_model: Optional[BaseEstimator] = None,
    ):
        """Initialize a GPTClassifier.

        Args:
            property_name (str): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            extractor (ClassificationExtractor, optional): Callable object that can extract
                integers from the completions produced by the querier.
                Defaults to ClassificationExtractor().
        """
        self.property_name = property_name
        self.tuner = tuner
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )
        self.extractor = extractor
        self.formatter = ClassificationFormatter(
            representation_column="repr_with_ngram",
            label_column="prop",
            property_name=property_name,
            num_classes=None,
        )
        self.model_name = None
        self.tune_res = None
        self.count_vectorizer = CountVectorizer() if count_vectorizer is None else count_vectorizer
        self.ngram_model = MultinomialNB() if ngram_model is None else ngram_model

    def _fit_ngram_model(self, X: ArrayLike, y: ArrayLike):
        X = self.count_vectorizer.fit_transform(X)
        self.ngram_model.fit(X, y)

    def _predict_ngram_model(self, X: ArrayLike):
        X = self.count_vectorizer.transform(X)
        return self.ngram_model.predict(X)

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
        """
        df = self._prepare_df(X, y)
        self._fit_ngram_model(X, y)
        ngram_preds = self._predict_ngram_model(X)
        df["ngram_preds"] = ngram_preds
        df["repr_with_ngram"] = (
            df["repr"] + " with n-gram prediction " + df["ngram_preds"].astype(str)
        )
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))

        ngram_preds = self._predict_ngram_model(X)
        df["ngram_preds"] = ngram_preds
        df["repr_with_ngram"] = (
            df["repr"] + " with n-gram prediction " + df["ngram_preds"].astype(str)
        )
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        return extracted


class DifficultNGramClassifier:
    """Highlight cases an N-Gram model struggles with."""

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        count_vectorizer: Optional[CountVectorizer] = None,
        ngram_model: Optional[BaseEstimator] = None,
    ):
        """Initialize a GPTClassifier.

        Args:
            property_name (str): Name of the property to be predicted.
               This will be part of the prompt.
            tuner (Tuner): Tuner object to be used for fine tuning.
               This specifies the model to be used and the fine-tuning settings.
            querier_settings (Optional[dict], optional): Settings for the querier.
                Defaults to None.
            extractor (ClassificationExtractor, optional): Callable object that can extract
                integers from the completions produced by the querier.
                Defaults to ClassificationExtractor().
        """
        self.property_name = property_name
        self.tuner = tuner
        self.querier_setting = (
            querier_settings if querier_settings is not None else {"max_tokens": 3}
        )
        self.extractor = extractor
        self.formatter = ClassificationFormatter(
            representation_column="repr_with_ngram",
            label_column="prop",
            property_name=property_name,
            num_classes=None,
        )
        self.model_name = None
        self.tune_res = None
        self.count_vectorizer = CountVectorizer() if count_vectorizer is None else count_vectorizer
        self.ngram_model = MultinomialNB() if ngram_model is None else ngram_model

    def _fit_ngram_model(self, X: ArrayLike, y: ArrayLike):
        X = self.count_vectorizer.fit_transform(X)
        self.ngram_model.fit(X, y)

    def _predict_ngram_model(self, X: ArrayLike):
        X = self.count_vectorizer.transform(X)
        return self.ngram_model.predict(X)

    def _prepare_df(self, X: ArrayLike, y: ArrayLike):
        rows = []
        for i in range(len(X)):
            rows.append({"repr": X[i], "prop": y[i]})
        return pd.DataFrame(rows)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fine tune a GPT-3 model on a dataset.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)
            y (ArrayLike): Target data (typically array of property values)
        """
        df = self._prepare_df(X, y)
        self._fit_ngram_model(X, y)
        ngram_preds = self._predict_ngram_model(X)
        df["ngram_incorrect"] = ngram_preds != y
        n_gram_diffcult = [
            "This was a difficult example. Pay attention." if x else ""
            for x in df["ngram_incorrect"]
        ]
        df["repr_with_ngram"] = df["repr"] + " " + n_gram_diffcult
        formatted = self.formatter(df)
        tune_res = self.tuner(formatted)
        self.model_name = tune_res["model_name"]
        self.tune_res = tune_res

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict property values for a set of molecular representations.

        Args:
            X (ArrayLike): Input data (typically array of molecular representations)

        Returns:
            ArrayLike: Predicted property values
        """
        df = self._prepare_df(X, [0] * len(X))

        df["repr_with_ngram"] = df["repr"] + " " + [""] * len(X)
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        return extracted


class MultiRepGPTClassifier(GPTClassifier):
    """GPT Classifier trained on muliple representations."""

    def __init__(
        self,
        property_name: str,
        tuner: Tuner,
        querier_settings: Optional[dict] = None,
        extractor: ClassificationExtractor = ClassificationExtractor(),
        rep_names: Optional[List[str]] = None,
    ) -> None:
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
        self.rep_names = rep_names

    def _prepare_df(self, X: ArrayLike, y: ArrayLike, shuffle: bool = True):
        # assumes that columns in X are the different representations
        rows = []
        for i in range(len(X)):
            for j in range(len(X[i])):
                repr_name = self.rep_names[j] + " " if self.rep_names is not None else ""
                rows.append({"repr": repr_name + X[i][j], "prop": y[i], "mol": i, "rep": j})

        if shuffle:
            return pd.DataFrame(rows).sample(frac=1)
        return pd.DataFrame(rows)

    def _predict(self, X: ArrayLike) -> ArrayLike:
        df = self._prepare_df(X, [0] * len(X), shuffle=False)
        formatted = self.formatter(df)
        querier = Querier(self.model_name, **self.querier_setting)
        completions = querier(formatted)
        extracted = self.extractor(completions)
        # reshape such that predictions also have multiple columns
        # one per representation and one row per molecule
        # we can get the molecule and the representation from the df
        predictions = np.zeros((len(X), len(X[0])))

        for i in range(len(X)):
            for rep in range(len(X[0])):
                subset = df[(df["mol"] == i) & (df["rep"] == rep)]
                predictions[i, rep] = extracted[subset.index[0]]

        return predictions

    def predict(self, X: ArrayLike, return_std: bool = False):
        predictions = self._predict(X)
        if return_std:
            return np.mean(predictions, axis=1), np.std(predictions, axis=1)
        return np.mean(predictions, axis=1)
