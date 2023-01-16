from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.cv import train_test_cv_classification_baseline
from gptchem.data import get_moosavi_cv_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

train_sizes = [10, 20, 50, 100]
num_classes = [2, 5]
test_size = 100
representation = ["grouped_mof", "mofid"]


def train_test(train_size, representation, num_class, seed=42):
    data = get_moosavi_cv_data()
    formatter = ClassificationFormatter(
        representation_column=representation,
        label_column="Cv_gravimetric_300.00",
        property_name="heat capacity",
        num_classes=num_class,
    )
    formatted = formatter(data)

    ts = min(test_size, len(data) - train_size)
    train, test = train_test_split(
        formatted,
        train_size=train_size,
        test_size=ts,
        stratify=formatted["label"],
        random_state=seed,
    )

    baseline_res = train_test_cv_classification_baseline(
        data,
        train_mofid=train["representation"],
        test_mofid=test["representation"],
        formatter=formatter,
        repr_col=representation,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test, logprobs=num_class)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test["label"], extracted)

    print(
        f"Ran train size {train_size} and got accuracy {gpt_metrics['accuracy']}, XGB baseline {baseline_res['accuracy']}"
    )

    summary = {
        "num_classes": num_class,
        "num_train_points": train_size,
        **gpt_metrics,
        "baseline": baseline_res,
        "representation": representation,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for rep in representation:
        for i in range(10):
            for train_size in train_sizes:
                for num_class in num_classes:
                    train_test(train_size, rep, num_class, i + 104567)
