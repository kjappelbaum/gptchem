from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from gauche.dataloader import ReactionLoader

from gptchem.evaluator import evaluate_classification, get_regression_metrics

from ..models.gpr import GPRBaseline

models = [
    ("ohe", "tanimoto"),
    ("rxnfp", "linear"),
    ("rxnfp", "rbf"),
    ("drfp", "linear"),
    ("drfp", "rbf"),
]


def train_test_rxn_classification_baseline(ds, train_data, test_data, formatter):
    res = {}

    for fp, kernel in models:
        with TemporaryDirectory() as tmpdir:
            name_app = "RXN" if fp in ("rxnfp", "drfp") else ""
            joined_data = pd.concat([train_data, test_data])
            joined_data.to_csv(Path(tmpdir) / "data.csv", index=False)
            in_train_mask = joined_data["rxn"].isin(train_data["rxn"])
            loader = ReactionLoader()
            loader.load_benchmark(ds + name_app, Path(tmpdir) / "data.csv")
            loader.featurize(fp)

            features = loader.features.astype(float)
            labels = loader.labels
            train_features, test_features = features[in_train_mask], features[~in_train_mask]
            train_labels, test_labels = labels[in_train_mask], labels[~in_train_mask]

            gpr = GPRBaseline(kernel=kernel)
            gpr.fit(train_features, train_labels)
            predictions = gpr.predict(test_features).flatten()
            res[f"{fp}-{kernel}"] = predictions

    target_label_bin = formatter.bin(test_data["yield"])
    classification_res = {}

    for k, v in res.items():
        classification_res[k] = evaluate_classification(target_label_bin, formatter.bin(v))

    return {"predictions": res, "metrics": classification_res}


def train_test_rxn_regressions_baseline(ds, train_data, test_data):
    res = {}

    for fp, kernel in models:
        with TemporaryDirectory() as tmpdir:
            name_app = "RXN" if fp in ("rxnfp", "drfp") else ""
            joined_data = pd.concat([train_data, test_data])
            joined_data.to_csv(Path(tmpdir) / "data.csv", index=False)
            in_train_mask = joined_data["rxn"].isin(train_data["rxn"])
            loader = ReactionLoader()
            loader.load_benchmark(ds + name_app, Path(tmpdir) / "data.csv")
            loader.featurize(fp)

            features = loader.features.astype(float)
            labels = loader.labels
            train_features, test_features = features[in_train_mask], features[~in_train_mask]
            train_labels, test_labels = labels[in_train_mask], labels[~in_train_mask]

            gpr = GPRBaseline(kernel=kernel)
            gpr.fit(train_features, train_labels)
            predictions = gpr.predict(test_features).flatten()
            res[f"{fp}-{kernel}"] = predictions

    regression_res = {}

    for k, v in res.items():
        regression_res[k] = get_regression_metrics(test_labels, v)

    return {"predictions": res, "metrics": regression_res}
