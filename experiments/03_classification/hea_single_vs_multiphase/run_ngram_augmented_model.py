import os
import time
from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.data import get_hea_phase_data
from gptchem.evaluator import evaluate_classification
from gptchem.gpt_classifier import NGramGPTClassifier
from gptchem.tuner import Tuner

NUM_REPEATS = 5
LEARNING_CURVE_POINTS = [10, 20, 50, 100, 200][::-1]
TEST_SIZE = 250

outdir = "ngram_augmented"

if not os.path.exists(outdir):
    os.makedirs(outdir)


def train_test_model(num_train_points, seed):
    data = get_hea_phase_data()

    train, test = train_test_split(
        data,
        train_size=num_train_points,
        test_size=TEST_SIZE,
        stratify=data["phase_binary_encoded"],
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    classifier = NGramGPTClassifier("phase", tuner=tuner)
    classifier.fit(train["Alloy"].values, train["phase_binary_encoded"].values)
    predictions = classifier.predict(test["Alloy"].values)
    gpt_metrics = evaluate_classification(test["phase_binary_encoded"].values, predictions)

    print(f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}")

    summary = {
        **gpt_metrics,
        "train_size": num_train_points,
        "predictions": predictions,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_pickle(Path(outdir) / f"{timestr}-{num_train_points}-{seed}-summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(NUM_REPEATS):
        for num_train_point in LEARNING_CURVE_POINTS:
            train_test_model(num_train_point, seed+30)
