import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification, get_regression_metrics

from .randomforest import RFClassificationBaseline
from ..fingerprints.mol_fingerprints import compute_fragprints, compute_morgan_fingerprints
from ..models.gpr import GPRBaseline
from ..models.xgboost import XGBClassificationBaseline

OPV_FEATURES = [f"ecpf_{i}" for i in range(1064)]


def train_test_opv_classification_baseline(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    formatter,
    seed: int = 42,
    num_trials: int = 100,
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

    train["bin"] = formatter.bin(train[label_column])
    test["bin"] = formatter.bin(test[label_column])

    # tabpfn
    print("Computing morgan fingerprints...")
    X_train = compute_morgan_fingerprints(train["SMILES"].values, n_bits=100)
    X_test = compute_morgan_fingerprints(test["SMILES"].values, n_bits=100)
    classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    classifier.fit(X_train, train["bin"].values)
    predicted_bins, _ = classifier.predict(X_test, return_winning_probability=True)
    tabpfn_results = {
        "true_bins": test["bin"],
        "predicted_bins": predicted_bins,
        **evaluate_classification(test["bin"].astype(int).values, predicted_bins.astype(int)),
    }

    X_train = compute_fragprints(train["SMILES"].values)
    X_test = compute_fragprints(test["SMILES"].values)
    baseline = GPRBaseline()
    baseline.fit(X_train, train[label_column].values)

    predictions = baseline.predict(X_test)

    # we clip as out-of-bound predictions result in NaNs
    pred = np.clip(predictions.flatten(), a_min=formatter.bins[0], a_max=formatter.bins[-1])
    predicted_bins = formatter.bin(pred)
    gpr_results = {
        "true_bins": test["bin"],
        "predicted_bins": predicted_bins,
        **evaluate_classification(test["bin"].astype(int).values, predicted_bins.astype(int)),
    }

    X_train, y_train = train[OPV_FEATURES], train["bin"]
    X_test, y_test = test[OPV_FEATURES], test["bin"]

    xgb = XGBClassificationBaseline(seed=seed, num_trials=num_trials)
    xgb.tune(X_train, y_train)
    xgb.fit(X_train, y_train)
    predictions = xgb.predict(X_test)

    xgb_results = {
        "true_bins": test["bin"],
        "predicted_bins": predictions,
        **evaluate_classification(test["bin"].astype(int).values, predictions.astype(int)),
    }

    X_train, y_train = train[OPV_FEATURES], train["bin"]
    X_test, y_test = test[OPV_FEATURES], test["bin"]

    rf = RFClassificationBaseline(seed=seed, num_trials=num_trials)
    rf.tune(X_train, y_train)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    rf_results = {
        "true_bins": test["bin"],
        "predicted_bins": predictions,
        **evaluate_classification(test["bin"].astype(int).values, predictions.astype(int)),
    }

    return {"tabpfn": tabpfn_results, "gpr": gpr_results, "xgb": xgb_results, "rf": rf_results}


def train_test_opv_regression_baseline(
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
