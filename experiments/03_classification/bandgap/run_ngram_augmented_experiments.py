import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.xtras import save_pickle
from loguru import logger
from sklearn.model_selection import train_test_split

from gptchem.baselines.bandgap import train_test_bandgap_classification_baseline
from gptchem.data import get_qmug_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.gpt_classifier import NGramGPTClassifier
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_classes = [2, 5]
num_training_points = [10, 50, 100, 200, 500, 1000]  # 1000
representations = ["SMILES", "SELFIES", "InChI"]
num_test_points = 250
num_repeats = 10

outdir = "ngram_augmented_balanced"
if not Path(outdir).exists():
    Path(outdir).mkdir()

def train_test_model(num_classes, representation, num_train_points, seed):
    data = get_qmug_data()
    # formatter = ClassificationFormatter(
    #     representation_column=representation,
    #     property_name="HOMO-LUMO gap",
    #     label_column="DFT_HOMO_LUMO_GAP_mean_ev",
    #     num_classes=num_classes,
    # )

    data["bin"] = pd.qcut(
        data["DFT_HOMO_LUMO_GAP_mean_ev"], num_classes, labels=np.arange(num_classes)
    ).astype(int)

    train, test = train_test_split(
        data,
        train_size=num_train_points,
        test_size=num_test_points,
        stratify=data["bin"],
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    classifier = NGramGPTClassifier("HOMO-LUMO gap", tuner=tuner)
    classifier.fit(train[representation].values, train["bin"].values)
    predictions = classifier.predict(test[representation].values)
    gpt_metrics = evaluate_classification(test["bin"].values, predictions)

    print(f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}")

    summary = {
        **gpt_metrics,
        "train_size": num_train_points,
        "num_classes": num_classes,
        "predictions": predictions,
        "representation": representation,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_pickle(Path(outdir) / f"{timestr}-summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_classes in num_classes:
            for num_train_point in num_training_points:
                for representation in representations:
                    try:
                        train_test_model(num_classes, representation, num_train_point, i)
                    except Exception as e:
                        logger.exception(e)
