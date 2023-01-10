import pytest
from gptchem.data import get_qmug_data
from gptchem.formatter import ClassificationFormatter
from gptchem.baselines.bandgap import train_test_bandgap_classification_baseline

@pytest.mark.slow 
@pytest.mark.parametrize('tabpfn', (True, False))
def test_train_test_bandgap_classification_baseline(tabpfn):
    data = get_qmug_data()
    formatter = ClassificationFormatter(
        representation_column="SMILES",
        property_name="bandgap",
        label_column='DFT_HOMO_LUMO_GAP_mean_ev', 
        num_classes=2,
    )
    res = train_test_bandgap_classification_baseline(
        data, train_size=100, test_size=10, formatter=formatter, num_trials=10, tabpfn=tabpfn)
    assert res["accuracy"] > 0.2