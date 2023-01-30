from pathlib import Path

from loguru import logger
from fastcore.all import L
from fastcore.xtras import save_pickle
from gptchem.tuner import Tuner
from gptchem.gpt_classifier import GPTClassifier
from gptchem.data import get_matbench_glass
from sklearn.model_selection import train_test_split
from automatminer import MatPipe
from gptchem.evaluator import evaluate_classification


num_repeats = 10
num_train_points = [10, 50, 100, 200, 500, 1000]

logger.enable("gptchem")

def train_test(train_size, seed): 
    data = get_matbench_glass()
    train, test = train_test_split(data, train_size=train_size, random_state=seed, stratify=data['gfa'])

    try:
        pipe = MatPipe.from_preset("express")

        pipe.fit(train, 'gfa')

        predictions = pipe.predict(test)

        baseline_metrics = evaluate_classification(predictions, test)   

    except Exception:
        baseline_metrics = {
            'accuracy': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'roc_auc': None,
            'pr_auc': None,
        }

    classifier = GPTClassifier('glass formation ability', Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False))

    classifier.fit(train['composition'].values, train['gfa'].values)

    predictions = classifier.predict(test['composition'].values)

    gpt_metrics = evaluate_classification(predictions, test)

    res = {
        'baseline': baseline_metrics,
        **gpt_metrics,
        'train_size': train_size,
    }

    save_pickle(Path(classifier.tune_res["outdir"]) / "summary.pkl", res)

    return res


if __name__ == "__main__":
    for i in range(num_repeats):
        for train_size in num_train_points:
            train_test(train_size, i)