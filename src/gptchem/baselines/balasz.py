import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification

from ..models.xgboost import XGBClassificationBaseline


def train_test_mof_synthesizability_baseline(train, test, target_column, seed=42):
    joined = pd.concat([train, test])
    train = train.copy()
    test = test.copy()

    ligand = joined["ligand name"].unique()
    inorganic_salt = joined["inorganic salt"].unique()
    additional = joined["additional"].unique()
    solvent_1 = joined["solvent1"].unique()
    solvent_2 = joined["solvent2"].unique()
    solvent_3 = joined["solvent3"].unique()

    categorizer_ligand = LabelEncoder()
    categorizer_ligand.fit(ligand)
    categorizer_inorganic_salt = LabelEncoder()
    categorizer_inorganic_salt.fit(inorganic_salt)
    categorizer_additional = LabelEncoder()
    categorizer_additional.fit(additional)
    categorizer_solvent_1 = LabelEncoder()
    categorizer_solvent_1.fit(solvent_1)
    categorizer_solvent_2 = LabelEncoder()
    categorizer_solvent_2.fit(solvent_2)
    categorizer_solvent_3 = LabelEncoder()
    categorizer_solvent_3.fit(solvent_3)

    train["ligand name"] = categorizer_ligand.transform(train["ligand name"])
    train["inorganic salt"] = categorizer_inorganic_salt.transform(train["inorganic salt"])
    train["additional"] = categorizer_additional.transform(train["additional"])
    train["solvent1"] = categorizer_solvent_1.transform(train["solvent1"])
    train["solvent2"] = categorizer_solvent_2.transform(train["solvent2"])
    train["solvent3"] = categorizer_solvent_3.transform(train["solvent3"])

    test["ligand name"] = categorizer_ligand.transform(test["ligand name"])
    test["inorganic salt"] = categorizer_inorganic_salt.transform(test["inorganic salt"])
    test["additional"] = categorizer_additional.transform(test["additional"])
    test["solvent1"] = categorizer_solvent_1.transform(test["solvent1"])
    test["solvent2"] = categorizer_solvent_2.transform(test["solvent2"])
    test["solvent3"] = categorizer_solvent_3.transform(test["solvent3"])

    X_train = train[
        [
            "ligand name",
            "inorganic salt",
            "additional",
            "solvent1",
            "solvent2",
            "solvent3",
            "V/V solvent1 [ ]",
            "V/V solvent2 [ ]",
            "V/V solvent3 [ ]",
            "T [°C]",
            "t [h]",
        ]
    ]
    y_train = train[target_column]

    X_test = test[
        [
            "ligand name",
            "inorganic salt",
            "additional",
            "solvent1",
            "solvent2",
            "solvent3",
            "V/V solvent1 [ ]",
            "V/V solvent2 [ ]",
            "V/V solvent3 [ ]",
            "T [°C]",
            "t [h]",
        ]
    ]
    y_test = test[target_column]

    tabpfn = TabPFNClassifier()
    tabpfn.fit(X_train, y_train)
    y_pred = tabpfn.predict(X_test)

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    xgb = XGBClassificationBaseline(seed=seed)
    xgb.tune(X_train, y_train)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    results = {
        "tabpfn_metrics": evaluate_classification(y_test, y_pred),
        "dummy_metrics": evaluate_classification(y_test, y_pred_dummy),
        "xgb_metrics": evaluate_classification(y_test, y_pred_xgb),
    }

    return results
