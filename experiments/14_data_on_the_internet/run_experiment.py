from gptchem.tuner import Tuner
from gptchem.gpt_classifier import GPTClassifier
from gptchem.evaluator import evaluate_classification
from fastcore.xtras import save_pickle
import time
import pandas as pd
import openai
from gptchem.data import get_hea_phase_data
import dotenv

dotenv.load_dotenv("../../.env", override=True)
import os


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

    results = evaluate_classification(test_frame["phase_binary_encoded"].values, predictions)

    report = {
        "train_size": train_size,
        "seed": seed,
        **results,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_pickle(f"reports/{timestr}-report.pkl", report)


if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]
    for seed in range(5):
        for train_size in [10, 20, 50, 100, 200, 500]:
            run_experiments(train_size, seed)
