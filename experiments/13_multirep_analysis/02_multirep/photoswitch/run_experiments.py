from pathlib import Path
import time
import pandas as pd
import numpy as np
from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.photoswitch import train_test_photoswitch_classification_baseline
from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.gpt_classifier import MultiRepGPTClassifier
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from loguru import logger

num_classes = [2, 5][::-1]
num_training_points = [10, 20, 50, 100, 200]  # 1000
representations = [
    ["selfies_x", "deepsmiles", "canonical", "inchi_x", "tucan", "iupac_name"],
    ["selfies_x", "deepsmiles", "canonical", "inchi_x", "tucan"],
    ["selfies_x", "deepsmiles", "canonical"],
    ["selfies_x", "canonical"],
][::-1]
max_num_test_points = 100
num_repeats = 10

outdir = "results_balanced"
if not Path(outdir).exists():
    Path(outdir).mkdir()


def train_test_model(num_classes, representation, num_train_points, seed):
    data = pd.read_csv("../photoswitch_data.csv")
    data = data.dropna(subset=["E isomer pi-pi* wavelength in nm"] + representation)
    data["binned"] = pd.qcut(
        data["E isomer pi-pi* wavelength in nm"], num_classes, labels=np.arange(num_classes)
    )

    num_test_points = min((max_num_test_points, len(data) - num_train_points))

    train, test = train_test_split(
        data,
        train_size=num_train_points,
        test_size=num_test_points,
        stratify=data["binned"],
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    classifier = MultiRepGPTClassifier("transition wavelength", tuner, rep_names=representation)

    classifier.fit(train[representation].values, train["binned"].values)

    predictions = classifier._predict(test[representation].values).astype(int)

    predictions_mode = np.array([np.argmax(np.bincount(pred)) for pred in predictions.astype(int)])
    predictions_var = np.array([np.var(pred) for pred in predictions.astype(int)])

    confident = predictions_var < 0.1

    cm_all = evaluate_classification(test["binned"].values, predictions_mode)
    cm_confident = evaluate_classification(
        test["binned"].values[confident], predictions_mode[confident]
    )

    print(
        f"Ran train size {num_train_points} and got accuracy {cm_all['accuracy']:.2f} and confident accuracy {cm_confident['accuracy']:.2f}"
    )

    summary = {
        "num_classes": num_classes,
        "num_train_points": num_train_points,
        "predictions": predictions,
        "predictions_mode": predictions_mode,
        "predictions_var": predictions_var,
        "confident": confident,
        "cm_all": cm_all,
        "cm_confident": cm_confident,
        "representation": representation,
        "num_test_points": num_test_points,
        "train_len": len(train),
        "test_len": len(test),
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_pickle(Path(outdir) / f"{timestamp}_summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_class in num_classes:
            for num_train_points in num_training_points:
                for representation in representations:
                    try:
                        train_test_model(num_class, representation, num_train_points, i + 674567)
                    except Exception as e:
                        logger.exception(e)
