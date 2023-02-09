from crabnet.crabnet_ import CrabNet

from gptchem.data import get_hea_phase_data
from sklearn.model_selection import train_test_split
from pymatgen.core import Composition
import numpy as np
from pathlib import Path
from fastcore.xtras import save_pickle
from gptchem.evaluator import evaluate_classification
from loguru import logger
import gc
from typing import Dict, Callable
from collections import OrderedDict
import fire

logger.enable("gptchem")

NUM_REPEATS = 10
LEARNING_CURVE_POINTS = [20, 50, 100, 200, 10,]
TEST_SIZE = 250

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
    data = get_hea_phase_data()

    data['formula'] = data['Alloy'].map(lambda x: try_except_nan(comp_to_str, x))
    data['target'] = data['phase_binary_encoded']
    data.dropna(inplace=True)
    data = data[['formula', 'target']]
    train_data, test_data = train_test_split(data, test_size=train_size, random_state=seed, stratify=data['target'])
    cb = CrabNet(mat_prop="phase_binary_encoded", classification=True, learningcurve=False, losscurve=False, save=False)
    cb.classification = True
    cb.optimizer = None
    cb = CrabNet(mat_prop="phase_binary_encoded", classification=True, learningcurve=False, losscurve=False, save=False)
    cb.classification = True
    cb.fit(train_data)
    predictions = (cb.predict(test_data) > 0.5).astype(int)
    res = evaluate_classification(test_data['target'].values, predictions)
    res['train_size'] = train_size
    logger.info(f"Train size: {train_size},  accuracy: {res['accuracy']}")
    save_pickle(Path(OURDIR) / f"summary-{train_size}-{seed}.pkl", res)
    cb.optimizer = None

if __name__ == "__main__":
    fire.Fire(train_test_evaluate)