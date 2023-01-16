import pytest

from gptchem.baselines.solubility import train_test_solubility_classification_baseline
from gptchem.data import get_esol_data, get_solubility_test_data
from gptchem.formatter import ClassificationFormatter


def test_train_test_solubility_classification_baseline():
    train_data = get_esol_data()
    test_data = get_solubility_test_data().sample(10)

    formatter = ClassificationFormatter(
        representation_column="SMILES",
        property_name="solubility",
        label_column="measured log(solubility:mol/L)",
        num_classes=2,
    )

    formatted = formatter(train_data)
    train_data = train_data.sample(10)

    res = train_test_solubility_classification_baseline(
        train_data, test_data, formatter=formatter, num_trials=10
    )

    assert res["xgb"]["accuracy"] > 0.5
