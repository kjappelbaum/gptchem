import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from gptchem.data import get_moosavi_pcv_data
from gptchem.evaluator import evaluate_classification

xgb_hyperparams = {
    "xgb__reg_lambda": 0.049238826317067365,
    "xgb__reg_alpha": 0.049238826317067365,
    "xgb__n_estimators": 300,
    "xgb__max_depth": 10,
    "xgb__learning_rate": 0.1,
}

TARGET_p = "pCv_300.00"
TARGET = "Cv_gravimetric_300.00"

FEATURES = [
    "site Number",
    "site AtomicWeight",
    "site Row",
    "site Column",
    "site Electronegativity",
    "site CovalentRadius",
    "AGNI eta=8.00e-01",
    "AGNI eta=1.23e+00",
    "AGNI eta=1.88e+00",
    "AGNI eta=2.89e+00",
    "AGNI eta=4.43e+00",
    "AGNI eta=6.80e+00",
    "AGNI eta=1.04e+01",
    "AGNI eta=1.60e+01",
    "G2_0.05",
    "G2_4.0",
    "G2_20.0",
    "G2_80.0",
    "G4_0.005_1.0_1.0",
    "G4_0.005_1.0_-1.0",
    "G4_0.005_4.0_1.0",
    "G4_0.005_4.0_-1.0",
    "local difference in Number",
    "local difference in AtomicWeight",
    "local difference in Row",
    "local difference in Column",
    "local difference in Electronegativity",
    "local difference in CovalentRadius",
    "local signed difference in Number",
    "local signed difference in AtomicWeight",
    "local signed difference in Row",
    "local signed difference in Column",
    "local signed difference in Electronegativity",
    "local signed difference in CovalentRadius",
    "maximum local difference in Number",
    "maximum local difference in AtomicWeight",
    "maximum local difference in Row",
    "maximum local difference in Column",
    "maximum local difference in Electronegativity",
    "maximum local difference in CovalentRadius",
    "minimum local difference in Number",
    "minimum local difference in AtomicWeight",
    "minimum local difference in Row",
    "minimum local difference in Column",
    "minimum local difference in Electronegativity",
    "minimum local difference in CovalentRadius",
]


def xgb_model(data, feat, target, best_params=xgb_hyperparams):
    pipe_xgb = Pipeline(
        [
            ("scaling", StandardScaler()),
            ("variance_threshold", VarianceThreshold(threshold=0.95)),
            ("xgb", XGBRegressor()),
        ]
    )
    pipe_xgb.set_params(xgb__reg_lambda=best_params["xgb__reg_lambda"])
    pipe_xgb.set_params(xgb__reg_alpha=best_params["xgb__reg_alpha"])
    pipe_xgb.set_params(xgb__n_estimators=best_params["xgb__n_estimators"])
    pipe_xgb.set_params(xgb__max_depth=best_params["xgb__max_depth"])
    pipe_xgb.set_params(xgb__learning_rate=best_params["xgb__learning_rate"])

    pipe_xgb.fit(data[feat], data[target])

    return pipe_xgb


def train_xgb_ensemble(data, feat, target, best_params=xgb_hyperparams, num_rounds: int = 10):
    datasets = [data.sample(len(data), replace=True) for _ in range(num_rounds)]
    models = [xgb_model(d, feat, target, best_params) for d in datasets]
    return models


def predict_xgb_ensemble(models, data, feat, identifier_col, identifiers):
    structure_pred = []
    for structure in identifiers:
        structure_data = data[data[identifier_col] == structure]
        pred = (
            np.array([m.predict(structure_data[feat]) for m in models]).sum(axis=1)
            / structure_data["site AtomicWeight"].sum()
        )
        structure_pred.append(pred.mean())

    return np.array(structure_pred)


def train_test_cv_classification_baseline(
    cv_data, train_mofid, test_mofid, formatter, num_rounds=10, repr_col="mofid"
):
    data = get_moosavi_pcv_data()
    train_data = data[data[repr_col].isin(train_mofid)]
    test_data = data[data[repr_col].isin(test_mofid)]
    assert len(test_data[repr_col].unique()) == len(
        test_mofid
    ), f"test data ({len(test_data[repr_col].unique())}) and test mofid ({len(test_mofid)}) must have the same length"
    test_cv_data = cv_data[cv_data[repr_col].isin(test_mofid)]

    assert len(test_cv_data) == len(
        test_data[repr_col].unique()
    ), f"test data ({len(test_data[repr_col].unique())}) and test cv data ({len(test_cv_data)}) must have the same length"
    models = train_xgb_ensemble(train_data, FEATURES, TARGET_p, num_rounds=num_rounds)
    pred = predict_xgb_ensemble(models, test_data, FEATURES, repr_col, test_cv_data[repr_col])
    assert len(pred) == len(test_cv_data)

    binned = formatter.bin(pred).astype(int)
    return {
        "pred": pred,
        "binned": binned,
        "test_mofid": test_mofid,
        **evaluate_classification(formatter.bin(test_cv_data[TARGET]), binned),
    }
