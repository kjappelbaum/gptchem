import pytest

from gptchem.baselines.freesolv import train_test_freesolv_regression_baseline
from gptchem.data import get_freesolv_data
from gptchem.formatter import RegressionFormatter


def test_train_test_freesolv_regression_baseline():
    data = get_freesolv_data()
    formatter = RegressionFormatter(
        representation_column="smiles",
        property_name="hydration free energy",
        label_column="expt",
    )

    smiles = data["smiles"].tolist()
    train_smiles = smiles[:10]
    test_smiles = smiles[10:20]
    res = train_test_freesolv_regression_baseline(
        data, train_smiles=train_smiles, test_smiles=test_smiles, formatter=formatter
    )

    assert res["mean_absolute_error"] < 3
