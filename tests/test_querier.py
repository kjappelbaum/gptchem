import pandas as pd
import pytest

from gptchem.querier import Querier


@pytest.mark.requires_api_key
@pytest.mark.parametrize("temperature", [0, 2.0])
def test_querier(temperature):
    querier = Querier("ada")
    df = pd.DataFrame(
        {"prompt": ["What is the HOMO-LUMO gap of CCCC?###"], "completion": [" 0.0@@@"]}
    )

    completions = querier.query(df, temperature=temperature)
    assert isinstance(completions, dict)
    assert isinstance(completions["choices"], list)
    assert isinstance(completions["logprobs"], list)
    assert isinstance(completions["choices"][0], str)

    df_many = pd.DataFrame(
        {
            "prompt": [
                "What is the HOMO-LUMO gap of CCCC?###",
                "What is the HOMO-LUMO gap of [O]?###",
            ],
            "completion": [" 0.0@@@", " 0.0@@@"],
        }
    )

    completions = querier.query(df_many)
    assert isinstance(completions, dict)
    assert isinstance(completions["choices"], list)
    assert len(completions["choices"]) == 2
