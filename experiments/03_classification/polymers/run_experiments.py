from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.polymer import train_test_polymer_classification_baseline
from gptchem.data import get_polymer_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_classes = [2, 5]
num_training_points = [10, 20, 50, 100, 200, 500]
num_test_points = 250
num_repeats = 10


def train_test_model(num_classes, num_train_points, seed):
    data = get_polymer_data()
    formatter = ClassificationFormatter(
        representation_column="string",
        property_name="adsorption energy",
        label_column="deltaGmin",
        num_classes=num_classes,
    )
    xgboost_baseline = train_test_polymer_classification_baseline(
        data,
        train_size=num_train_points,
        test_size=num_test_points,
        formatter=formatter,
        tabpfn=False,
        seed=seed,
    )
    tabpfn_baseline = train_test_polymer_classification_baseline(
        data,
        train_size=num_train_points,
        test_size=num_test_points,
        formatter=formatter,
        tabpfn=True,
        seed=seed,
    )

    formatted = formatter(data)
    train, test = train_test_split(
        formatted,
        train_size=num_train_points,
        test_size=num_test_points,
        stratify=formatted["label"],
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test, logprobs=num_classes)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test["label"], extracted)

    print(
        f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}, XGB baseline {xgboost_baseline['accuracy']}, and TabPFN baseline {tabpfn_baseline['accuracy']}"
    )

    summary = {
        **gpt_metrics,
        "xgboost_baseline": xgboost_baseline,
        "tabpfn_baseline": tabpfn_baseline,
        "train_size": num_train_points,
        "num_classes": num_classes,
        "completions": completions,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_classes in num_classes:
            for num_train_points in num_training_points:
                try:
                    train_test_model(num_classes, num_train_points, i+1456450)
                except Exception as e:
                    print(e)
