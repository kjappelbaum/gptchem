import os

import pytest

from gptchem.tuner import Tuner


@pytest.mark.requires_api_key
@pytest.mark.slow
def test_tuner(get_prompts, tmp_path):
    tuner = Tuner("ada", outdir=tmp_path)
    result = tuner.tune(get_prompts)

    assert isinstance(result, dict)

    assert "ft" in result["ft_id"]
    assert "ada" in result["model_name"]

    assert os.path.exists(result["outdir"])

    assert os.path.exists(os.path.join(result["outdir"], "train.jsonl"))

    assert os.path.exists(os.path.join(result["outdir"], "summary.json"))
