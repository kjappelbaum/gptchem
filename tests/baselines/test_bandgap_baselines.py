import pytest

from gptchem.baselines.bandgap import (
    train_test_bandgap_classification_baseline,
    train_test_bandgap_regression_baseline,
)
from gptchem.data import get_qmug_data
from gptchem.formatter import ClassificationFormatter, RegressionFormatter


@pytest.mark.slow
@pytest.mark.parametrize("tabpfn", (True, False))
def test_train_test_bandgap_classification_baseline(tabpfn):
    data = get_qmug_data()
    formatter = ClassificationFormatter(
        representation_column="SMILES",
        property_name="bandgap",
        label_column="DFT_HOMO_LUMO_GAP_mean_ev",
        num_classes=2,
    )
    res = train_test_bandgap_classification_baseline(
        data, train_size=100, test_size=10, formatter=formatter, tabpfn=tabpfn
    )
    assert res["accuracy"] > 0.2


@pytest.mark.slow
def test_train_test_bandgap_regression_baseline():
    data = get_qmug_data()
    formatter = RegressionFormatter(
        representation_column="SMILES",
        property_name="bandgap",
        label_column="DFT_HOMO_LUMO_GAP_mean_ev",
    )

    smiles = data["SMILES"].tolist()
    train_smiles = smiles[:10]
    test_smiles = smiles[10:20]
    res = train_test_bandgap_regression_baseline(
        data, train_smiles=train_smiles, test_smiles=test_smiles, formatter=formatter
    )

    assert res["mean_absolute_error"] < 1.0
