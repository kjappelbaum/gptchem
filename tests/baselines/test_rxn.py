import pytest
from gptchem.data import get_doyle_rxn_data
from gptchem.baselines.rxn import train_test_rxn_classification_baseline
from sklearn.model_selection import train_test_split
from gptchem.formatter import ReactionClassificationFormatter

@pytest.mark.slow
def test_train_test_rxn_classification_baseline():
    data = get_doyle_rxn_data()

    formatter = ReactionClassificationFormatter.from_preset(
        'DreherDoyle', 2
    )
    formatted = formatter(data)
    train_data, test_data = train_test_split(data, train_size=20, test_size=10)
    res = train_test_rxn_classification_baseline('DreherDoyle', train_data, test_data, formatter)
    assert "metrics" in res
    assert res["metrics"]['ohe-tanimoto']["accuracy"] >= .3
    assert res["metrics"]["drfp-rbf"]["accuracy"] >= .5