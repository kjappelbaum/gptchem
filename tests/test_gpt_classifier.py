import pandas as pd

from gptchem.gpt_classifier import GPTClassifier
from gptchem.tuner import Tuner


def test_gpt_classifier():
    classifier = GPTClassifier(
        property_name="transition wavelength",
        tuner=Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False),
    )

    df = classifier._prepare_df(["CC", "CDDFSS"], [0, 1])

    assert isinstance(df, pd.DataFrame)
