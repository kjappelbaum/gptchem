import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

num_train_points = [10, 20, 50, 100, 200, 500][::-1]
num_classes = [5, 2]
representations = ["SMILES", "SELFIES", "InChI"]

import os
import time
from pathlib import Path

import numpy as np
from fastcore.xtras import save_pickle

from gptchem.data import get_esol_data, get_solubility_test_data
from gptchem.evaluator import evaluate_classification

num_train_points = [10, 20, 50, 100, 200, 500][::-1]
num_classes = [5, 2]
representations = ["SMILES", "SELFIES", "InChI"]

num_repeats = 10
outdir = "ngram_baseline"

if not os.path.exists(outdir):
    os.makedirs(outdir)


def train_test_model(num_classes, representation, num_train_points, seed):
    data = get_esol_data()
    data = data.dropna(subset=["measured log(solubility:mol/L)"])
    train_subset = data.sample(n=num_train_points, random_state=seed)
    train_subset = train_subset.reset_index(drop=True)
    train_subset["binned"] = (
        train_subset["measured log(solubility:mol/L)"]
        > np.median(data["measured log(solubility:mol/L)"])
    ).astype(int)
    test = get_solubility_test_data()
    test["binned"] = (
        test["measured log(solubility:mol/L)"] > np.median(data["measured log(solubility:mol/L)"])
    ).astype(int)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_subset[representation])
    X_test = vectorizer.transform(test[representation])

    clf = MultinomialNB()
    clf.fit(X_train, train_subset["binned"])

    y_pred = clf.predict(X_test)
    y_true = test["binned"]

    metrics = evaluate_classification(y_true, y_pred)

    print(f"Ran train size {num_train_points} and got accuracy {metrics['accuracy']:.3f}")

    summary = {
        **metrics,
        "train_size": num_train_points,
        "num_classes": num_classes,
        "representation": representation,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_pickle(
        Path(os.path.join(outdir, f"{timestr}_{num_train_points}_{representation}.pkl")), summary
    )


if __name__ == "__main__":
    for num_class in num_classes:
        for representation in representations:
            for num_train_point in num_train_points:
                for seed in range(num_repeats):
                    try:
                        train_test_model(num_class, representation, num_train_point, seed)
                    except Exception as e:
                        print(e)
                        continue
