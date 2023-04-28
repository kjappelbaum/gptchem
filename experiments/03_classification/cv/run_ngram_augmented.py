import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.data import get_moosavi_cv_data
from gptchem.evaluator import evaluate_classification
from gptchem.gpt_classifier import NGramGPTClassifier
from gptchem.tuner import Tuner

NUM_REPEATS = 5
LEARNING_CURVE_POINTS = [10, 20, 50, 100]
TEST_SIZE = 100

representations = ["grouped_mof", "mofid", "composition"][::-1]

outdir = "ngram_augmented"

if not os.path.exists(outdir):
    os.makedirs(outdir)


def train_test_model(num_train_points, representation, num_classes, seed):
    data = get_moosavi_cv_data()
    data = data.dropna(subset=["Cv_gravimetric_300.00"])
    data["binned"] = pd.qcut(
        data["Cv_gravimetric_300.00"], num_classes, labels=np.arange(num_classes)
    )

    train, test = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min(TEST_SIZE, len(data) - num_train_points),
        stratify=data["binned"],
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    classifier = NGramGPTClassifier("heat capacity", tuner=tuner)
    classifier.fit(train[representation].values, train["binned"].values)
    predictions = classifier.predict(test[representation].values)
    gpt_metrics = evaluate_classification(test["binned"].values, predictions)

    print(
        f"Ran train size {num_train_points} and repr {representation} and got accuracy {gpt_metrics['accuracy']}"
    )

    summary = {
        **gpt_metrics,
        "train_size": num_train_points,
        "predictions": predictions,
        "representation": representation,
        "num_classes": num_classes,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_pickle(
        Path(outdir)
        / f"{timestr}-{num_train_points}-{num_classes}_{representation}_{seed}-summary.pkl",
        summary,
    )


if __name__ == "__main__":
    for num_classes in [2, 5]:
        for seed in range(NUM_REPEATS):
            for num_train_point in LEARNING_CURVE_POINTS:
                for representation in representations:
                    train_test_model(num_train_point, representation, num_classes, seed)
