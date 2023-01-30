from gptchem.data import get_photoswitch_data
from gptchem.gpt_regressor import GPTRegressor


def test_gpt_regression():
    """Test GPT-3 regression."""
    data = get_photoswitch_data()
    gpt = GPTRegressor("pce", tuner=None)
    data = gpt._prepare_df(data["SMILES"], data["E isomer pi-pi* wavelength in nm"])
    assert data.shape == (403, 2)
