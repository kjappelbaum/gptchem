import os

import pandas as pd
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def get_photoswitch_data():
    return pd.read_csv(os.path.join(_THIS_DIR, "test_files", "photoswitches.csv"))
