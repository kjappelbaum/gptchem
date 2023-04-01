from typing import Optional

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
