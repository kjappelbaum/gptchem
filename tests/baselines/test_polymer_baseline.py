import pytest
from gptchem.baselines.polymer import train_test_polymer_classification_baseline
from gptchem.data import get_polymer_data
from gptchem.formatter import ClassificationFormatter


@pytest.mark.slow
@pytest.mark.parametrize('tabpfn', (True, False))
def test_train_test_polymer_classification_baseline(tabpfn):
    data = get_polymer_data()
    formatter = ClassificationFormatter(
        representation_column="string",
        property_name="adsorption energy",
        label_column="deltaGmin",
        num_classes=2,
    )
    res = train_test_polymer_classification_baseline(
        data, train_size=100, test_size=10, formatter=formatter, num_trials=10, tabpfn=tabpfn
    )
    assert res["accuracy"] > 0.5
