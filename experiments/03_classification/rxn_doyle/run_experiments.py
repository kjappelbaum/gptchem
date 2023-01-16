from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.rxn import train_test_rxn_classification_baseline
from gptchem.data import get_doyle_rxn_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ReactionClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

train_sizes = [10, 20, 50, 100, 200]
num_classes = [5,2]
num_test_points = 100
use_one_hot = [True, False]


def train_test(train_size, num_class, seed, one_hot):
    data = get_doyle_rxn_data()
    formatter = ReactionClassificationFormatter.from_preset(
        "DreherDoyle", num_classes=num_class, one_hot=one_hot
    )

    formatted = formatter(data)

    train_idx, test_idx = train_test_split(
        data.index,
        train_size=train_size,
        test_size=num_test_points,
        stratify=formatted["label"],
        random_state=seed,
    )

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_formatted = formatted.iloc[train_idx]
    test_formatted = formatted.iloc[test_idx]

    baseline = train_test_rxn_classification_baseline(
        "DreherDoyle", train_data=train_data, test_data=test_data, formatter=formatter
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train_formatted)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test_formatted, logprobs=num_class)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test_formatted["label"], extracted)

    print(
        f"Train size: {train_size}, Accuracy: {gpt_metrics['accuracy']}, Baseline RXNFP-RBF: {baseline['metrics']['rxnfp-rbf']['accuracy']}"
    )

    summary = {
        "num_classes": num_class,
        "one_hot": one_hot,
        "train_size": train_size,
        **gpt_metrics,
        "baseline": baseline,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(10):
        for oh in use_one_hot:
            for train_size in train_sizes:
                for num_class in num_classes:
                    train_test(train_size, num_class, i+644456656, one_hot=oh)
