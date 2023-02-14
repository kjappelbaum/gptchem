# https://github.com/PatWalters/solubility/blob/master/solubility_comparison.py
import itertools

import deepchem
from deepchem.models import GraphConvModel, WeaveModel
from deepchem.models.sklearn_models import SklearnModel
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification, get_regression_metrics

from .esol import ESOLCalculator
from ..fingerprints.mol_fingerprints import compute_fragprints, compute_morgan_fingerprints
from ..models.gpr import GPRBaseline
from ..models.xgboost import XGBClassificationBaseline, XGBRegressionBaseline


def featurize_data(tasks, featurizer, normalize, df, smiles_col="SMILES"):
    loader = deepchem.data.InMemoryLoader(tasks=tasks, featurizer=featurizer)
    smiles = df[smiles_col]
    dataset = loader.create_dataset(smiles, shard_size=8192)
    move_mean = True
    if normalize:
        transformers = [
            deepchem.trans.NormalizationTransformer(
                transform_y=True, dataset=dataset, move_mean=move_mean
            )
        ]
    else:
        transformers = []
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset, featurizer, transformers


def generate_prediction(df, model, featurizer, transformers, smiles_col="SMILES"):
    mol_list = [Chem.MolFromSmiles(x) for x in df[smiles_col]]
    val_feats = featurizer.featurize(mol_list)
    res = model.predict(deepchem.data.NumpyDataset(val_feats), transformers)
    # kind of a hack
    # seems like some models return a list of lists and others (e.g. RF) return a list
    # check to see if the first element in the returned array is a list, if so, flatten the list
    if type(res[0]) is list:
        df["pred_vals"] = list(itertools.chain.from_iterable(*res))
    else:
        df["pred_vals"] = res

    return df["pred_vals"]


def generate_graph_conv_model():
    batch_size = 128
    model = GraphConvModel(1, batch_size=batch_size, mode="regression")
    return model


def generate_weave_model():
    batch_size = 64
    model = WeaveModel(
        1, batch_size=batch_size, learning_rate=1e-3, use_queue=False, mode="regression"
    )
    return model


def generate_rf_model():
    sklearn_model = RandomForestRegressor(n_estimators=500)
    return SklearnModel(sklearn_model)


def calc_esol(df, smiles_col="SMILES"):
    esol_calculator = ESOLCalculator()
    res = []
    for smi in df[smiles_col].values:
        mol = Chem.MolFromSmiles(smi)
        res.append(esol_calculator.calc_esol(mol))
    return res


def run_model(
    model_func, task_list, featurizer, normalize, train_df, test_df, nb_epoch, smiles_col="SMILES"
):
    dataset, featurizer, transformers = featurize_data(
        task_list, featurizer, normalize, train_df, smiles_col=smiles_col
    )
    model = model_func()
    if nb_epoch > 0:
        model.fit(dataset, nb_epoch)
    else:
        model.fit(dataset)
    pred_df = generate_prediction(test_df, model, featurizer, transformers)
    return pred_df


def solubility_baseline(
    df_train, df_test, smiles_col="SMILES", task_name="measured log(solubility:mol/L)"
):
    task_list = [task_name]

    featurizer = deepchem.feat.WeaveFeaturizer()
    model_func = generate_weave_model
    weave_res = run_model(model_func, task_list, featurizer, True, df_train, df_test, nb_epoch=30)

    featurizer = deepchem.feat.ConvMolFeaturizer()
    model_func = generate_graph_conv_model
    gc_res = run_model(model_func, task_list, featurizer, True, df_train, df_test, nb_epoch=20)

    gpr = GPRBaseline()
    X_train = compute_fragprints(df_train["SMILES"].values)
    X_test = compute_fragprints(df_test["SMILES"].values)
    y_train = df_train[task_name].values
    y_test = df_test[task_name].values

    esol = calc_esol(df_test)

    gpr.fit(X_train, y_train)
    gpr_res = gpr.predict(X_test)

    res = {
        "true": df_test[task_name],
        "esol": esol,
        "weave": weave_res.values,
        "graph_conv": gc_res.values,
        "gpr": gpr_res.flatten(),
    }

    return res


def train_test_solubility_classification_baseline(
    df_train,
    df_test,
    formatter,
    smiles_col="SMILES",
    task_name="measured log(solubility:mol/L)",
    seed=42,
    num_trials=100,
):
    baselines = solubility_baseline(df_train, df_test, smiles_col=smiles_col, task_name=task_name)

    binned_weave_res = formatter.bin(baselines["weave"])
    binned_graph_conv_res = formatter.bin(baselines["graph_conv"])
    binned_gpr_res = formatter.bin(baselines["gpr"])
    binned_esol = formatter.bin(baselines["esol"])
    binned_true = formatter.bin(baselines["true"])

    y_train = formatter.bin(df_train[task_name].values)
    tabpfnclassifier = TabPFNClassifier()
    X_train = compute_morgan_fingerprints(df_train["SMILES"].values, n_bits=100)
    X_test = compute_morgan_fingerprints(df_test["SMILES"].values, n_bits=100)
    tabpfnclassifier.fit(X_train, y_train)
    tabpfn_res = tabpfnclassifier.predict(X_test)
    tabpfn_res = formatter.bin(tabpfn_res)
    binned_tabpfn_res = formatter.bin(tabpfn_res)

    X_train = compute_fragprints(df_train["SMILES"].values)
    X_test = compute_fragprints(df_test["SMILES"].values)
    xgbclassifier = XGBClassificationBaseline(seed=42, num_trials=num_trials)
    xgbclassifier.tune(X_train, y_train)
    xgbclassifier.fit(X_train, y_train)
    predictions = xgbclassifier.predict(X_test)

    res = {
        "true": binned_true,
        "weave": evaluate_classification(binned_true, binned_weave_res),
        "graph_conv": evaluate_classification(binned_true, binned_graph_conv_res),
        "gpr": evaluate_classification(binned_true, binned_gpr_res),
        "tabpfn": evaluate_classification(binned_true, binned_tabpfn_res),
        "xgb": evaluate_classification(binned_true, predictions),
        "esol": evaluate_classification(binned_true, binned_esol),
    }

    return res


def train_test_solubility_regression_baseline(
    df_train,
    df_test,
    smiles_col="SMILES",
    task_name="measured log(solubility:mol/L)",
    num_trials=100,
):
    baselines = solubility_baseline(df_train, df_test, smiles_col=smiles_col, task_name=task_name)

    y_train = df_train[task_name]
    y_test = df_test[task_name]
    X_train = compute_morgan_fingerprints(df_train["SMILES"].values, n_bits=100)
    X_test = compute_morgan_fingerprints(df_test["SMILES"].values, n_bits=100)

    X_train = compute_fragprints(df_train["SMILES"].values)
    X_test = compute_fragprints(df_test["SMILES"].values)
    # xgbclassifier = XGBRegressionBaseline(seed=42, num_trials=num_trials)
    # xgbclassifier.tune(X_train, y_train)
    # xgbclassifier.fit(X_train, y_train)
    # predictions = xgbclassifier.predict(X_test)

    res = {
        "true": y_train,
        "weave": get_regression_metrics(y_test, baselines["weave"]),
        "graph_conv": get_regression_metrics(y_test, baselines["graph_conv"]),
        "gpr": get_regression_metrics(y_test, baselines["gpr"]),
        #  "xgb": get_regression_metrics(y_test, predictions),
        "esol": get_regression_metrics(y_test, baselines["esol"]),
    }

    return res
