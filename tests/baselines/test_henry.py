from gptchem.baselines.henry import train_test_henry_classification_baseline
from gptchem.data import get_moosavi_mof_data
from gptchem.formatter import ClassificationFormatter

from sklearn.model_selection import train_test_split
import pytest 

@pytest.mark.slow
def test_train_test_henry_classification_baseline():
    data = get_moosavi_mof_data()
    formatter = ClassificationFormatter(
        representation_column="mofid",
        property_name="logKH_CH4",
        label_column="logKH_CH4",
        num_classes=2,
    )
    formatted = formatter(data)
    train, test = train_test_split(data, train_size=100, test_size=10, random_state=42)

    res = train_test_henry_classification_baseline(
        train_set=train,
        test_set=test,
        formatter=formatter,
        target_col="logKH_CH4",
        num_trials=10,
        seed=42
    )

    assert res["xgb_metrics"]["accuracy"] > 0.5