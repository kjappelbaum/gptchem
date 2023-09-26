import pandas as pd 
from sklearn.model_selection import train_test_split
from gptchem.gpt_classifier import GPTClassifier
from gptchem.tuner import Tuner
import os
import time
from fastcore.xtras import save_pickle
from gptchem.evaluator import evaluate_classification
from pathlib import Path

outdir = "results_new2"
if not os.path.exists(outdir):
    os.makedirs(outdir)

def test(train_size, seed, balanced=True): 
    data = pd.read_pickle('merged_data.pkl')

    # select half of class 1 and half of class 2
    if balanced:
        train, test = train_test_split(
        data,
        test_size=200,
        stratify=data["oxidation_state"],
        random_state=seed,
    )

        half_size = train_size // 2
        train_1 = train[train["oxidation_state"] == 1].sample(half_size, random_state=seed)
        train_2 = train[train["oxidation_state"] == 2].sample(half_size, random_state=seed)
    else:
        train, test = train_test_split(
        data,
        train_size=train_size,
        test_size=200,
        stratify=data["oxidation_state"],
        random_state=seed,
    )
        train_1 = train[train["oxidation_state"] == 1]
        train_2 = train[train["oxidation_state"] == 2]

    print(train["oxidation_state"].value_counts())

    train = pd.concat([train_1, train_2])

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    classifier = GPTClassifier("oxidation states", tuner=tuner)
    classifier.fit(train["composition"].values, train["oxidation_state"].values)
    predictions = classifier.predict(test["composition"].values)
    gpt_metrics = evaluate_classification(test["oxidation_state"].values, predictions)

    bv_metrics = evaluate_classification(test["oxidation_state"].values, test["bv"].values)
    oximachine_metrics = evaluate_classification(test["oxidation_state"].values, test["prediction"].values)

    print(f"Ran train size {train_size} and got kappa {gpt_metrics['kappa']}, {bv_metrics['kappa']}, {oximachine_metrics['kappa']}")

    timestr = time.strftime("%Y%m%d-%H%M%S")

    summary = {
        **gpt_metrics,
        'bv': bv_metrics,
        'oximachine': oximachine_metrics,
        "train_size": train_size,
        "predictions": predictions,
        "balanced": balanced,
    }

    save_pickle(Path(outdir) / f"{timestr}-{train_size}-{seed}-summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(3):
        for balanced in [False]:
            for train_size in [4000, 5000, 6000, 7000]:
                try:
                    test(train_size, seed+45, balanced=balanced)
                except Exception as e:
                    print(e)
                    pass    