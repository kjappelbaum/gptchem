from gptchem.data import get_water_stability
from gptchem.baselines.water_stability import train_test_waterstability_baseline
from gptchem.gpt_classifier import GPTClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from fastcore.xtras import save_pickle
from gptchem.evaluator import evaluate_classification
from gptchem.tuner import Tuner 

num_rounds = 10
train_sizes = [10, 20, 50, 100, 150]


def train_test(train_size, seed):
    data = get_water_stability()
    train, test = train_test_split(
        data, train_size=train_size, stratify=data["stability_int"], random_state=seed
    )

    baseline = train_test_waterstability_baseline(train, test, seed=seed)

    classifier = GPTClassifier(property_name="water stability", tuner= Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False))
    classifier.fit(train["normalized_names"].values, train["stability_int"].values)
    predictions = classifier.predict(test["normalized_names"].values)
    metrics = evaluate_classification(test["stability_int"].values, predictions)

    summary = {
        "train_size": train_size,
        "predictions": predictions,
        **baseline,
        **classifier.tune_res,
        **metrics,
    }

    save_pickle(Path(classifier.tune_res["outdir"]) / "summary.pkl", summary)

    return summary

if __name__ == '__main__':
    for i in range(num_rounds):
        for train_size in train_sizes:
            train_test(train_size=train_size, seed=i + 13)