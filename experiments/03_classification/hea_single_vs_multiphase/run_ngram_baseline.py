import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.xtras import save_pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from gptchem.data import get_hea_phase_data
from gptchem.evaluator import evaluate_classification

train_sizes = [10, 20, 50, 100, 200][::-1]
TEST_SIZE = 250
num_repeats = 10
outdir = "ngram_baseline"

if not os.path.exists(outdir):
    os.makedirs(outdir)


def train_test_model(num_train_points, seed):
    data = get_hea_phase_data()

    train, test = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min((TEST_SIZE, len(data) - num_train_points)),
        stratify=data["phase_binary_encoded"],
        random_state=seed,
    )
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train["Alloy"])
    X_test = vectorizer.transform(test["Alloy"])

    clf = MultinomialNB()
    clf.fit(X_train, train["phase_binary_encoded"])

    y_pred = clf.predict(X_test)
    y_true = test["phase_binary_encoded"]

    metrics = evaluate_classification(y_true, y_pred)

    print(f"Ran train size {num_train_points} and got accuracy {metrics['accuracy']:.3f}")

    summary = {
        **metrics,
        "train_size": num_train_points,
        "representation": "Alloy",
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_pickle(Path(os.path.join(outdir, f"{timestr}_{seed}_{num_train_points}.pkl")), summary)


if __name__ == "__main__":
    for num_train_point in train_sizes:
        for seed in range(num_repeats):
            train_test_model(num_train_point, seed)
