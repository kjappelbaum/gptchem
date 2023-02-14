import pytest
from sklearn.model_selection import train_test_split

from gptchem.baselines.water_stability import train_test_waterstability_baseline
from gptchem.data import get_water_stability


@pytest.mark.slow
def test_train_test_waterstability_baseline():
    data = get_water_stability()

    train, test = train_test_split(
        data, train_size=10, test_size=10, stratify=data["stability_int"]
    )

    res = train_test_waterstability_baseline(train, test, num_trials=10)

    assert isinstance(res, dict)

    assert res["xgboost"]["accuracy"] > 0.3
    assert res["tabpfn"]["accuracy"] > 0.3
