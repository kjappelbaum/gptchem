import pytest

from gptchem.baselines.opv import train_test_opv_classification_baseline
from gptchem.data import get_opv_data
from gptchem.formatter import ClassificationFormatter


@pytest.mark.slow
def test_train_test_opv_classification_baseline():
    data = get_opv_data()
    formatter = ClassificationFormatter(
        representation_column="SMILES",
        property_name="PCE",
        label_column="PCE_ave(%)",
        num_classes=2,
    )
    res = train_test_opv_classification_baseline(
        data,
        train_size=10,
        test_size=10,
        formatter=formatter,
        num_trials=5,
    )

    assert "tabpfn" in res
    assert "gpr" in res
    assert "xgb" in res
    assert "rf" in res

    assert res["rf"]["accuracy"] > 0.3
