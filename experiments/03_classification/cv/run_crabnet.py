import gc
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict

import fire
import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from fastcore.xtras import save_pickle
from loguru import logger
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split

from gptchem.data import get_moosavi_cv_data
from gptchem.evaluator import evaluate_classification

logger.enable("gptchem")

NUM_REPEATS = 10
LEARNING_CURVE_POINTS = [
    20,
    50,
    100,
    200,
    10,
]
TEST_SIZE = 100

_global_optimizer_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_optimizer_post_hooks: Dict[int, Callable] = OrderedDict()

OURDIR = "out-crabnet"
Path(OURDIR).mkdir(exist_ok=True)


def try_except_nan(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return np.nan


def comp_to_str(x):
    return str(Composition(x).reduced_formula)


def train_test_evaluate(train_size, seed):
    data = get_moosavi_cv_data()
    data["target"] = pd.qcut(data["Cv_gravimetric_300.00"], 2, labels=[0, 1])

    data["formula"] = data["composition"].map(lambda x: try_except_nan(comp_to_str, x))
    data.dropna(inplace=True)

    data.dropna(inplace=True)
    data = data[["formula", "target"]]
    train_data, test_data = train_test_split(
        data,
        train_size=train_size,
        test_size=min(TEST_SIZE, len(data) - train_size),
        random_state=seed,
        stratify=data["target"],
    )
    cb = CrabNet(
        mat_prop="target", classification=True, learningcurve=False, losscurve=False, save=False
    )
    cb.classification = True
    cb.optimizer = None
    cb = CrabNet(
        mat_prop="target", classification=True, learningcurve=False, losscurve=False, save=False
    )
    cb.classification = True
    cb.fit(train_data)
    predictions = (cb.predict(test_data) > 0.5).astype(int)
    res = evaluate_classification(test_data["target"].values, predictions)
    res["train_size"] = train_size

    logger.info(f"Train size: {train_size},  accuracy: {res['accuracy']}")
    save_pickle(Path(OURDIR) / f"summary-{train_size}-{seed}.pkl", res)
    cb.optimizer = None


if __name__ == "__main__":
    fire.Fire(train_test_evaluate)
