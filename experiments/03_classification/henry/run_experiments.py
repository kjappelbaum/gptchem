from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.henry import train_test_henry_classification_baseline
from gptchem.data import get_moosavi_mof_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

train_sizes = [10, 20, 50, 100, 200, 500]
num_classes = [5, 2]
targets = [
    ("logKH_CH4", "CH4 Henry coefficient"),
    ("logKH_CO2", "CO2 Henry coefficient"),
]

max_test_points = 250


def run_experiment(train_size, num_class, target, seed):
    data = get_moosavi_mof_data()
    target_col, target_name = target
    formatter = ClassificationFormatter(
        representation_column="mofid",
        property_name=target_name,
        label_column=target_col,
        num_classes=num_class,
    )
    formatted = formatter(data)
    train, test, train_formatted, test_formatted = train_test_split(
        data,
        formatted,
        train_size=train_size,
        test_size=max_test_points,
        stratify=formatted["label"],
        random_state=seed,
    )

    baseline_res = train_test_henry_classification_baseline(
        train_set=train,
        test_set=test,
        formatter=formatter,
        target_col=target_col,
        seed=seed,
        num_trials=100,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train_formatted)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test_formatted, logprobs=num_class)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test_formatted["label"], extracted)

    summary = {
        "train_size": train_size,
        "num_classes": num_class,
        "target": target,
        "baseline": baseline_res,
        "completions": completions,
        **gpt_metrics,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(10):
        for train_size in train_sizes:
            for num_class in num_classes:
                for target in targets:
                    run_experiment(train_size, num_class, target, i + 13552342)
