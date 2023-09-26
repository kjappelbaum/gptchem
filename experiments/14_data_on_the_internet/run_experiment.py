from dotenv import load_dotenv
from deepchem.molnet import load_tox21
import numpy as np
from gptchem.tuner import Tuner
from gptchem.gpt_classifier import GPTClassifier
from gptchem.evaluator import evaluate_classification
from fastcore.xtras import save_pickle
from imblearn.under_sampling import RandomUnderSampler
import time
import os
import openai
import pandas as pd

from gptchem.data import get_hea_phase_data


def run_experiments(train_size: int, seed: int):
    df = pd.read_csv("search_results.csv")
    df["Alloy"] = df["query"].replace('"', "", regex=True)
    test_alloys = df.query("state != 'Results for exact spelling'")["Alloy"].tolist()
    data = get_hea_phase_data()

    test_frame = data.query("Alloy in @test_alloys")

    train_frame = data.query("Alloy not in @test_alloys")

    train_subset = train_frame.sample(train_size)

    classifier = GPTClassifier("phase", tuner=Tuner(n_epochs=8))

    classifier.fit(train_subset["Alloy"].values, train_subset["phase_binary_encoded"].values)

    predictions = classifier.predict(test_frame["Alloy"].values)
