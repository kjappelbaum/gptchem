from gptchem.baselines.cv import train_test_cv_classification_baseline
from gptchem.data import get_moosavi_cv_data
from gptchem.formatter import ClassificationFormatter
import pytest

@pytest.mark.slow
def test_train_test_cv_classification_baseline():
    data = get_moosavi_cv_data()
    mofids= data["mofid"].unique()
    train_mofid = mofids[:10]
    test_mofid = mofids[10:20]

    formatter = ClassificationFormatter(
        representation_column='mofid', 
        property_name='cv',
        label_column='Cv_gravimetric_300.00',
        num_classes=2,
    )
    formatted = formatter(data)

    res = train_test_cv_classification_baseline(
        data,
        train_mofid=train_mofid,
        test_mofid=test_mofid,
        formatter=formatter)

    assert res['accuracy'] > 0.5