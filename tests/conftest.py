import os

import pandas as pd
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def get_photoswitch_data():
    return pd.read_csv(os.path.join(_THIS_DIR, "test_files", "photoswitches.csv"))


@pytest.fixture(scope="session")
def get_int_completion():
    return {"choices": [" 1"], "logprobs": [None], "model": "ada"}


@pytest.fixture(scope="session")
def get_float_completion():
    return {"choices": [" 1.2"], "logprobs": [None], "model": "ada"}


@pytest.fixture(scope="session")
def get_str_completion():
    return {"choices": [" CC#C@C"], "logprobs": [None], "model": "ada"}


@pytest.fixture(scope="session")
def get_prompts():
    return pd.read_csv(os.path.join(_THIS_DIR, "test_files", "prompts.csv"))
