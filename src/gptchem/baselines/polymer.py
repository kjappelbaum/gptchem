POLYMER_FEATURES = [
    "num_[W]",
    "max_[W]",
    "num_[Tr]",
    "max_[Tr]",
    "num_[Ta]",
    "max_[Ta]",
    "num_[R]",
    "max_[R]",
    "[W]",
    "[Tr]",
    "[Ta]",
    "[R]",
    "rel_shannon",
    "length",
]
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification, get_regression_metrics

from ..models.xgboost import XGBClassificationBaseline, XGBRegressionBaseline


def train_test_polymer_classification_baseline(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    formatter,
    tabpfn: bool = False,
    seed: int = 42,
    num_trials: int = 100,
):
    label_column = formatter.label_column
    df = df.dropna(subset=[formatter.label_column, formatter.representation_column])
    formatted = formatter(df)
    df["label"] = formatted["label"]
    train, test = train_test_split(
        df, train_size=train_size, test_size=test_size, stratify=df["label"], random_state=seed
    )

    X_train, y_train = train[POLYMER_FEATURES], train["label"]
    X_test, y_test = test[POLYMER_FEATURES], test["label"]

    if not tabpfn:
        baseline = XGBClassificationBaseline(num_trials=num_trials, seed=seed)
        baseline.tune(X_train, y_train)
        baseline.fit(X_train, y_train)

        predictions = baseline.predict(X_test)

    else:
        classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
        classifier.fit(X_train, y_train)
        predictions, _ = classifier.predict(X_test, return_winning_probability=True)

    return {
        "true_bins": y_test,
        "predicted_bins": predictions,
        "predictions": predictions if not tabpfn else None,
        **evaluate_classification(y_test, predictions),
    }


def train_test_polymer_regression_baseline(
    df: pd.DataFrame,
    train_smiles: List[str],
    test_smiles: List[str],
    formatter,
    seed: int = 42,
    num_trials: int = 100,
):
    label_column = formatter.label_column
    repr_column = formatter.representation_column
    df = df.dropna(subset=[formatter.label_column, formatter.representation_column])
    formatted = formatter(df)

    train = df[df[repr_column].isin(train_smiles)]
    test = df[df[repr_column].isin(test_smiles)]

    X_train, y_train = train[POLYMER_FEATURES], train[label_column]
    X_test, y_test = test[POLYMER_FEATURES], test[label_column]

    baseline = XGBRegressionBaseline(num_trials=num_trials, seed=seed)
    baseline.tune(X_train, y_train)
    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)

    return {
        "true": y_test,
        "predictions": predictions,
        **get_regression_metrics(y_test, predictions),
    }
