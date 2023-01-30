import pytest

from gptchem.baselines.balasz import train_test_mof_synthesizability_baseline
from gptchem.data import get_balasz_data


@pytest.mark.slow
def test_train_test_mof_synthesizability_baseline():
    data = get_balasz_data()
    data["success"] = data["score"] > 0.6
    data["success"] = data["success"].astype(int)
    train = data.iloc[:100]
    test = data.iloc[100:150]

    res = train_test_mof_synthesizability_baseline(train, test, "success")
    assert res["dummy_metrics"]["accuracy"] > 0.4
