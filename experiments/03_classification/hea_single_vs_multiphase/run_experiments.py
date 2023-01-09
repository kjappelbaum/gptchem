from typing import Optional
import pandas as pd

from sklearn.model_selection import train_test_split

from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.data import get_hea_phase_data
from pathlib import Path
from fastcore.xtras import save_pickle

NUM_REPEATS = 10
LEARNING_CURVE_POINTS = [10, 20, 50, 100, 200]
TEST_SIZE = 250


def train_test_evaluate(
    formatted: pd.DataFrame,
    train_size: int = 10,
    test_size: int = 10,
    basemodel: str = "ada",
    n_epochs: int = 8,
    learning_rate_multiplier: Optional[int] = 0.02,
    seed: int = 42,
) -> dict:
    train, test = train_test_split(
        formatted,
        train_size=train_size,
        test_size=test_size,
        stratify=formatted["label"],
        random_state=seed,
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

    print(f"Ran train size {train_size} and got accuracy {res['accuracy']}")
    save_pickle(Path(tune_summary["outdir"]) / "summary.pkl", summary)

    return summary


if __name__ == "__main__":
    for i in range(NUM_REPEATS):
        data = get_hea_phase_data()
        formatted = ClassificationFormatter(
            representation_column="Alloy",
            label_column="phase_binary_encoded",
            property_name="phase",
            num_classes=None,
            qcut=None,
        )(data)
        for train_size in LEARNING_CURVE_POINTS:
            try:
                train_test_evaluate(
                    formatted,
                    train_size=train_size,
                    test_size=TEST_SIZE,
                    basemodel="ada",
                    n_epochs=8,
                    learning_rate_multiplier=0.02,
                    seed=i,
                )
            except Exception as e:
                print(e)