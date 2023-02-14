import numpy as np
import pytest
from fastcore.foundation import L
from pycm import ConfusionMatrix

from gptchem.evaluator import (
    FrechetBenchmark,
    KLDivBenchmark,
    evaluate_classification,
    evaluate_generated_smiles,
    evaluate_photoswitch_smiles_pred,
    is_in_pubchem,
    is_valid_polymer,
    predict_photoswitch,
)


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


@pytest.mark.parametrize("input_output", [("C[N]1C=CC(=N1)N=NC2=CC=CC=C2", 310.0, 290.0)])
def test_predict_photoswitch(input_output):
    inp, e, z = input_output
    result = predict_photoswitch(inp)
    assert result[0] == pytest.approx(e, 10)
    assert result[1] == pytest.approx(z, 10)


def test_frechet_benchmark():
    fb = FrechetBenchmark(["CCC", "C[O]"], sample_size=2)
    res = fb.score(["CCC", "CCCCC"])
    assert len(res) == 2
    assert res[0] == pytest.approx(8.6, 0.3)
    assert res[1] == pytest.approx(0.2, 0.3)


def test_kl_div_benchmark():
    kld = KLDivBenchmark(["CCCC", "CCCCCC[O]", "CCCO"], 3)
    score = kld.score(["CCCCCCCCCCCCC", "CCC[O]", "CCCCC"])
    assert score == pytest.approx(0.55, 0.3)


def test_evaluate_generated_smiles():
    res = evaluate_generated_smiles(["CCC", "CCCCC"], ["CCC", "CCCCC"])
    assert res["frac_valid"] == 1.0
    assert res["frac_unique"] == 1.0
    assert res["frac_smiles_in_train"] == 1.0
    assert res["frac_smiles_in_pubchem"] == 1.0
    assert np.isnan(res["kld"])


def test_evaluate_photoswitch_smiles_pred():
    res = evaluate_photoswitch_smiles_pred(
        ["C[N]1C=CC(=N1)N=NC2=CC=CC=C2", "C[N]1C=CCCCC(=N1)N=NC2=CC=CC=C2"], [310, 290], [290, 310]
    )
    assert "e_pi_pi_star_metrics" in res
    assert "z_pi_pi_star_metrics" in res
    assert res["e_pi_pi_star_metrics"]["mean_absolute_error"] == pytest.approx(40, 1)


def test_is_in_pubchem():
    assert is_in_pubchem("CC")
    assert not is_in_pubchem("[N-]=[N+]=NCC(C=C1)=CC2=C1CCC3=C(/N=N\2)C=C(CN=[N+]=[N-])C=C3")


def test_is_valid_polymer():
    assert not is_valid_polymer("W8A!!A-B-R-A-A-B-W8B-R-R-B-5-R-6-W-W-R-R-W")
    assert not is_valid_polymer(
        "FR-BCB-B-W-A-A-W-A-B-R-A-A-W-W-A-W-A-A-B-B-W-R-A-ROA-A-R-R-A-A-R-A-R-R-R-B-W-R"
    )
    assert not is_valid_polymer("R-W-W-B-B-")
    assert not is_valid_polymer(
        "R-B-W-A-A-B-B-A-----W-A-W-W-R-R-R-R-R-R-W-R-R-A-A-B-R-R-W-W-A-B-R-A-R"
    )
    assert not is_valid_polymer("R.B.A.B.W.A.R.B.R.A.W.A.A.B.R.A.B.B")
