import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification, get_regression_metrics

from ..fingerprints.mol_fingerprints import compute_fragprints, compute_morgan_fingerprints
from ..models.gpr import GPRBaseline


def train_test_bandgap_classification_baseline(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    formatter,
    tabpfn: bool = False,
    seed: int = 42,
):
    label_column = formatter.label_column
    df = df.dropna(subset=[formatter.label_column, formatter.representation_column])
    formatted = formatter(df)
    train, test = train_test_split(
        df,
        train_size=train_size,
        test_size=test_size,
        stratify=formatted["label"],
        random_state=seed,
    )

    df_train = pd.DataFrame({"SMILES": train["SMILES"], "y": train[label_column]})
    df_test = pd.DataFrame({"SMILES": test["SMILES"], "y": test[label_column]})

    df_train["bin"] = formatter.bin(df_train["y"])
    df_test["bin"] = formatter.bin(df_test["y"])

    if tabpfn:
        X_train = compute_morgan_fingerprints(df_train["SMILES"].values, n_bits=100)
        X_test = compute_morgan_fingerprints(df_test["SMILES"].values, n_bits=100)
    else:
        X_train = compute_fragprints(df_train["SMILES"].values)
        X_test = compute_fragprints(df_test["SMILES"].values)

    if not tabpfn:
        baseline = GPRBaseline()
        baseline.fit(X_train, df_train["y"].values)

        predictions = baseline.predict(X_test)

        # we clip as out-of-bound predictions result in NaNs
        pred = np.clip(predictions.flatten(), a_min=formatter.bins[0], a_max=formatter.bins[-1])
        predicted_bins = formatter.bin(pred)

    else:
        classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
        classifier.fit(X_train, df_train["bin"].values)
        predicted_bins, _ = classifier.predict(X_test, return_winning_probability=True)

    return {
        "true_bins": df_test["bin"],
        "predicted_bins": predicted_bins,
        "predictions": predictions if not tabpfn else None,
        **evaluate_classification(df_test["bin"].astype(int).values, predicted_bins.astype(int)),
    }


def train_test_bandgap_regression_baseline(
    data,
    train_smiles,
    test_smiles,
    formatter,
):
    label_column = formatter.label_column
    data = data.dropna(subset=[formatter.label_column, formatter.representation_column])
    formatted = formatter(data)

    train = data[data["SMILES"].isin(train_smiles)]
    test = data[data["SMILES"].isin(test_smiles)]

    df_train = pd.DataFrame({"SMILES": train["SMILES"], "y": train[label_column]})
    df_test = pd.DataFrame({"SMILES": test["SMILES"], "y": test[label_column]})

    X_train = compute_fragprints(df_train["SMILES"].values)
    X_test = compute_fragprints(df_test["SMILES"].values)

    baseline = GPRBaseline()
    baseline.fit(X_train, df_train["y"].values)

    predictions = baseline.predict(X_test)

    return {
        "predictions": predictions,
        **get_regression_metrics(df_test["y"].values, predictions.flatten()),
    }
