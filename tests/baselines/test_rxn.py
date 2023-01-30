import pytest
from sklearn.model_selection import train_test_split

from gptchem.baselines.rxn import (
    train_test_rxn_classification_baseline,
    train_test_rxn_regressions_baseline,
)
from gptchem.data import get_doyle_rxn_data
from gptchem.formatter import ReactionClassificationFormatter


@pytest.mark.slow
def test_train_test_rxn_classification_baseline():
    data = get_doyle_rxn_data()

    formatter = ReactionClassificationFormatter.from_preset("DreherDoyle", 2)
    formatted = formatter(data)
    train_data, test_data = train_test_split(data, train_size=20, test_size=10)
    res = train_test_rxn_classification_baseline("DreherDoyle", train_data, test_data, formatter)
    assert "metrics" in res
    assert res["metrics"]["ohe-tanimoto"]["accuracy"] >= 0.3
    assert res["metrics"]["drfp-rbf"]["accuracy"] >= 0.5


@pytest.mark.slow
def test_rain_test_rxn_regressions_baseline():
    data = get_doyle_rxn_data()

    formatter = ReactionClassificationFormatter.from_preset("DreherDoyle", 2)
    formatted = formatter(data)
    train_data, test_data = train_test_split(data, train_size=20, test_size=10)
    res = train_test_rxn_regressions_baseline("DreherDoyle", train_data, test_data, formatter)
    assert "metrics" in res
    assert res["metrics"]["ohe-tanimoto"]["mean_absolute_error"] <= 30
    assert res["metrics"]["drfp-rbf"]["mean_absolute_error"] <= 30
