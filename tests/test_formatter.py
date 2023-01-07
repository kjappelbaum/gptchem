import pandas as pd
import pytest

from gptchem.formatter import ClassificationFormatter, RegressionFormatter


@pytest.mark.parametrize("qcut", [True, False])
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
