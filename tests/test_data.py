import pandas as pd

from gptchem.data import get_photoswitch_data


def test_get_photoswitch_data():
    df = get_photoswitch_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 300
    assert "SMILES" in df.columns
