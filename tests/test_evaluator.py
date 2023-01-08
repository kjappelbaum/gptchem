import numpy as np
import pytest
from fastcore.foundation import L
from pycm import ConfusionMatrix

from gptchem.evaluator import evaluate_classification


@pytest.mark.parametrize("container", [np.array, L])
def test_evaluate_classification(container):
    y = container([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = container([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    result = evaluate_classification(y, y_pred)
    assert result["frac_valid"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["acc_macro"] == 1.0
    assert result["racc"] == pytest.approx(0.1, 0.2)
    assert result["kappa"] == 1.0

    with pytest.raises(AssertionError):
        y_pred = container([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None])
        evaluate_classification(y, y_pred)

    y_pred = container([0, 1, 2, 3, 4, 5, 6, 7, 8, None])
    result = evaluate_classification(y, y_pred)
    assert result["frac_valid"] == 0.9

    y_pred = container([0, 1, 2, 3, 4, 5, 6, 7, 8, 9.0])
    result = evaluate_classification(y, y_pred)
    assert result["might_have_rounded_floats"] == False

    y_pred = container([0, 1, 2, 3, 4, 5, 6, 7, 8, 9.1])
    result = evaluate_classification(y, y_pred)
    assert result["might_have_rounded_floats"] == True

    assert isinstance(result["confusion_matrix"], ConfusionMatrix)
