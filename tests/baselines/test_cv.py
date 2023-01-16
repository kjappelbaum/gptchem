import pytest

from gptchem.baselines.cv import train_test_cv_classification_baseline
from gptchem.data import get_moosavi_cv_data
from gptchem.formatter import ClassificationFormatter


@pytest.mark.slow
@pytest.mark.parametrize("representation", ["mofid", "grouped_mof"])
def test_train_test_cv_classification_baseline(representation):
    data = get_moosavi_cv_data()
    mofids = data[representation].unique()
    train_mofid = mofids[:10]
    test_mofid = mofids[10:20]

    formatter = ClassificationFormatter(
        representation_column="mofid",
        property_name="cv",
        label_column="Cv_gravimetric_300.00",
        num_classes=2,
    )
    formatted = formatter(data)

    res = train_test_cv_classification_baseline(
        data,
        train_mofid=train_mofid,
        test_mofid=test_mofid,
        formatter=formatter,
        repr_col=representation,
    )

    assert res["accuracy"] > 0.5
