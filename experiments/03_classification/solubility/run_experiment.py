from gptchem.data import get_solubility_test_data, get_esol_data
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from gptchem.baselines.solubility import train_test_solubility_classification_baseline
from gptchem.extractor import ClassificationExtractor
from gptchem.evaluator import evaluate_classification

from fastcore.xtras import save_pickle
from pathlib import Path

num_train_points = [10, 20, 50, 100, 200, 500]
num_classes = [5,2]
representations = ['SMILES', 'SELFIES', 'InChI']


def train_test(train_size, representation, num_class, seed):
    test_data = get_solubility_test_data()
    train_data = get_esol_data()
  
    train_subset = train_data.sample(train_size)
    train_subset = train_subset.reset_index(drop=True)
    formatter = ClassificationFormatter(
        representation_column=representation,
        property_name="solubility",
        label_column="measured log(solubility:mol/L)",
        num_classes=num_class,
    )
    train_formatted = formatter(train_subset)
    test_formatted = formatter(test_data)
    baseline = train_test_solubility_classification_baseline(
        train_subset,
        test_data,
        formatter=formatter,
        seed=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train_formatted)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test_formatted, logprobs=num_class)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test_formatted["label"], extracted)

    print(
        f"Train size: {train_size}, Accuracy: {gpt_metrics['accuracy']}, Baseline ESOL: {baseline['esol']['accuracy']}"
    )

    res = {
        **gpt_metrics,
        **baseline,
        "train_size": train_size,
        "num_class": num_class,
        "representation": representation
    }

    save_pickle(Path(tune_res["outdir"]) / "results.pkl", res)

    return res


if __name__ == "__main__":
    for i in range(10):
        for train_size in num_train_points:
            for num_class in num_classes:
                for representation in representations:
                    train_test(train_size, representation, num_class, seed=i+54456726)
