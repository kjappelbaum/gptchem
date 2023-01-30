import pandas as pd
import pytest

from gptchem.data import get_doyle_rxn_data, get_photoswitch_data
from gptchem.formatter import (
    ClassificationFormatter,
    InverseDesignFormatter,
    ReactionClassificationFormatter,
    ReactionRegressionFormatter,
    RegressionFormatter,
)


@pytest.mark.parametrize("qcut", [True])  # ToDo: also test false
def test_classification_formatter(get_photoswitch_data, qcut):
    formatter = ClassificationFormatter(
        representation_column="SMILES",
        label_column="E isomer pi-pi* wavelength in nm",
        property_name="E isomer pi-pi* wavelength in nm",
        num_classes=5,
        qcut=qcut,
    )

    prompt = formatter.format_many(get_photoswitch_data)

    assert len(prompt) < len(get_photoswitch_data)
    assert isinstance(prompt, pd.DataFrame)

    assert (
        prompt["prompt"].iloc[0]
        == "What is the E isomer pi-pi* wavelength in nm of C[N]1C=CC(=N1)N=NC2=CC=CC=C2?###"
    )

    assert prompt["completion"].iloc[0] == " 0@@@"

    formatter = ClassificationFormatter(
        representation_column="SMILES",
        label_column="E isomer pi-pi* wavelength in nm",
        property_name="transition wavelength",
        num_classes=5,
        qcut=qcut,
    )

    prompt = formatter.format_many(get_photoswitch_data)

    assert len(prompt) < len(get_photoswitch_data)
    assert isinstance(prompt, pd.DataFrame)

    assert (
        prompt["prompt"].iloc[0]
        == "What is the transition wavelength of C[N]1C=CC(=N1)N=NC2=CC=CC=C2?###"
    )
    assert prompt["completion"].iloc[0] == " 0@@@"

    prompt = formatter(get_photoswitch_data)

    assert len(prompt) < len(get_photoswitch_data)
    assert isinstance(prompt, pd.DataFrame)

    assert (
        prompt["prompt"].iloc[0]
        == "What is the transition wavelength of C[N]1C=CC(=N1)N=NC2=CC=CC=C2?###"
    )
    assert prompt["completion"].iloc[0] == " 0@@@"


@pytest.mark.parametrize("num_digits", [1, 2, 3])
def test_regression_formatter(get_photoswitch_data, num_digits):
    formatter = RegressionFormatter(
        representation_column="SMILES",
        label_column="E isomer pi-pi* wavelength in nm",
        property_name="E isomer pi-pi* wavelength in nm",
        num_digits=num_digits,
    )

    prompt = formatter.format_many(get_photoswitch_data)

    assert len(prompt) < len(get_photoswitch_data)
    assert isinstance(prompt, pd.DataFrame)

    data_no_na = get_photoswitch_data.dropna(subset=["E isomer pi-pi* wavelength in nm", "SMILES"])
    for i, row in prompt.iterrows():
        assert row["prompt"].startswith("What is the E isomer pi-pi* wavelength in nm")
        assert row["prompt"].endswith("?###")
        assert row["completion"].startswith(" ")
        assert row["completion"].endswith("@@@")
        assert float(row["completion"].strip().replace("@@@", "")) == pytest.approx(
            float(data_no_na["E isomer pi-pi* wavelength in nm"].iloc[i]),
            rel=num_digits,
        )

    prompt = formatter(get_photoswitch_data)

    assert len(prompt) < len(get_photoswitch_data)
    assert isinstance(prompt, pd.DataFrame)


def test_reaction_classification_formatter():
    data = get_doyle_rxn_data()
    formatter = ReactionClassificationFormatter.from_preset("DreherDoyle", 2, one_hot=False)
    formatted = formatter(data)
    assert len(data) == len(formatted)
    assert len(formatted["label"].unique()) == 2

    formatter = ReactionClassificationFormatter.from_preset("DreherDoyle", 2, one_hot=True)
    formatted = formatter(data)
    assert len(data) == len(formatted)
    assert len(formatted["label"].unique()) == 2


def test_inverse_design_formatter():
    data = get_photoswitch_data()
    formatter = InverseDesignFormatter(
        representation_column="SMILES",
        property_columns=["E isomer pi-pi* wavelength in nm"],
        property_names=["E isomer pi-pi* wavelength in nm"],
    )
    formatted = formatter(data)
    assert len(data) >= len(formatted)


def test_reaction_regression_formatter():
    data = get_doyle_rxn_data()
    formatter = ReactionRegressionFormatter.from_preset("DreherDoyle", 0)
    formatted = formatter(data)
    assert len(data) == len(formatted)
    assert formatted["completion"].iloc[0] == " 70@@@"
