from pathlib import Path

from fastcore.xtras import save_pickle
from loguru import logger
from sklearn.model_selection import train_test_split

from gptchem.baselines.henry import train_test_henry_regression_baseline
from gptchem.data import get_moosavi_mof_data
from gptchem.evaluator import get_regression_metrics
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

train_sizes = [10, 20, 50, 100, 200, 500, 1000]
targets = [
    ("logKH_CH4", "CH4 Henry coefficient"),
    ("logKH_CO2", "CO2 Henry coefficient"),
]

max_test_points = 250


def run_experiment(train_size, target, seed):
    data = get_moosavi_mof_data()
    target_col, target_name = target
    formatter = RegressionFormatter(
        representation_column="mofid",
        property_name=target_name,
        label_column=target_col,
    )
    formatted = formatter(data)
    bin_col = data[target_col] > data[target_col].median()
    train, test, train_formatted, test_formatted = train_test_split(
        data,
        formatted,
        train_size=train_size,
        test_size=max_test_points,
        stratify=bin_col,
        random_state=seed,
    )

    baseline_res = train_test_henry_regression_baseline(
        train_set=train,
        test_set=test,
        formatter=formatter,
        seed=seed,
        num_trials=100,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train_formatted)
    querier = Querier(tune_res["model_name"])
    completions = querier(test_formatted)
    extractor = RegressionExtractor()
    extracted = extractor(completions)

    gpt_metrics = get_regression_metrics(test[target_col], extracted)

    summary = {
        "train_size": train_size,
        "target": target,
        "baseline": baseline_res,
        "completions": completions,
        **gpt_metrics,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(10):
        for train_size in train_sizes:
            for target in targets:
                try:
                    run_experiment(train_size, target, i + 22332642)
                except Exception as e:
                    logger.exception(e)
