import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

base_models = ["ada", "babbage", "curie", "davinci"]
train_sizes = [10, 50, 100, 200]
num_epochs = [1, 2, 4, 8]
learning_rate_multipliers = [0.02, 0.05, 0.1, 0.2]
num_trials = 5

TEST_SIZE = 100


def train_test_evaluate(
    formatted: pd.DataFrame,
    train_size: int = 10,
    test_size: int = 10,
    basemodel: str = "ada",
    n_epochs: int = 4,
    learning_rate_multiplier: Optional[int] = None, 
    seed: int = 42,
) -> dict:
    train, test = train_test_split(
        formatted, train_size=train_size, test_size=test_size, stratify=formatted["label"], random_state=seed
    )

    tuner = Tuner(
        base_model=basemodel,
        n_epochs=n_epochs,
        learning_rate_multiplier=learning_rate_multiplier,
        wandb_sync=False,
    )

    tune_summary = tuner(train)

    assert isinstance(tune_summary["model_name"], str)

    querier = Querier.from_preset(tune_summary["model_name"], preset="classification")

    completions = querier(test, logprobs=2)

    extractor = ClassificationExtractor()

    extracted = extractor(completions)

    res = evaluate_classification(test["label"], extracted)

    summary = {
        **tune_summary,
        **res,
        "completions": completions,
        "train_size": train_size,
        "test_size": test_size,
    }

    save_pickle(Path(tune_summary["outdir"]) / "summary.pkl", summary)

    return summary


def main():
    data = get_photoswitch_data()
    formatter = ClassificationFormatter(
    representation_column="SMILES",
    label_column="E isomer pi-pi* wavelength in nm",
    property_name="transition wavelength",
    num_classes=2,
    qcut=True,
)

    formatted = formatter(data)

    all_res = []
    for i in range(num_trials):
        for train_size in train_sizes:
            for basemodel in base_models:
                for n_epochs in num_epochs:
                    for learning_rate_multiplier in learning_rate_multipliers:
                        res = train_test_evaluate(
                            formatted,
                            train_size=train_size,
                            test_size=TEST_SIZE,
                            basemodel=basemodel,
                            n_epochs=n_epochs,
                            learning_rate_multiplier=learning_rate_multiplier,
                            seed=i,
                        )
                        all_res.append(res)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_pickle(f"{time_str}_all_res.pkl", all_res)

if __name__ == "__main__":
    main()
