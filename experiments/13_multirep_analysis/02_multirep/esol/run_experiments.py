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

num_classes = [2, 5][::-1]
num_training_points = [10, 50, 100, 200, 500, 1000]  # 1000
representations = [
    ["selfies", "deepsmiles", "canonical", "inchi", "tucan", "iupac_name"],
    ["selfies", "deepsmiles", "canonical", "inchi", "tucan"],
    ["selfies", "deepsmiles", "canonical"],
    ["selfies", "canonical"],
    ["canonical"],
]
max_num_test_points = 100
num_repeats = 10

outdir = "results_balanced"
if not Path(outdir).exists():
    Path(outdir).mkdir()


def train_test_model(num_classes, representation, num_train_points, seed):
    test_data = pd.read_csv("../../02_multirep/solubility_test_data.csv")
    train_data = pd.read_csv("../../02_multirep/esol_data.csv")

    test = test_data.dropna(subset=representation + ["measured log(solubility:mol/L)"])
    train = train_data.dropna(subset=representation + ["measured log(solubility:mol/L)"])

    # bin test and train using qcut, make sure we have the same bin edges
    # for both test and train. don't leak information from test to train
    # by using the same bin edges for both, return bins from qcut
    # and use them to bin the test data

    _, bins = pd.qcut(train["measured log(solubility:mol/L)"], num_classes, retbins=True)
    bins = [-np.inf, *bins[1:-1], np.inf]

    train["binned"] = pd.cut(
        train["measured log(solubility:mol/L)"],
        bins=bins,
        labels=np.arange(num_classes),
        include_lowest=True,
    )

    test["binned"] = pd.cut(
        test["measured log(solubility:mol/L)"],
        bins=bins,
        labels=np.arange(num_classes),
        include_lowest=True,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    classifier = MultiRepGPTClassifier("solubility", tuner, rep_names=representation)

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
        "train_len": len(train),
        "test_len": len(test_data),
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_pickle(Path(outdir) / f"{timestamp}_summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_class in num_classes:
            for num_train_points in num_training_points:
                for representation in representations:
                    try:
                        train_test_model(num_class, representation, num_train_points, i + 6879)
                    except Exception as e:
                        print(e)
