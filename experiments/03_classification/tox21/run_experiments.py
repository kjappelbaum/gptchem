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


def get_timestr():
    return time.strftime("%Y%m%d-%H%M%S")


load_dotenv("../../../.env", override=True)

openai.api_key = os.environ["OPENAI_API_KEY"]

name_mapping = {
    "NR-AR": "activity in the Androgen receptor, full length assay",
    "NR-AR-LBD": "activity in the Androgen receptor, ligand binding domain assay",
    "NR-AhR": "activity in the Aryl hydrocarbon receptor assay",
    "NR-Aromatase": "activity in the Aromatase assay",
    "NR-ER": "activity in the Estrogen receptor alpha, full length assay",
    "NR-ER-LBD": "activity in the Estrogen receptor alpha, LBD assay",
    "NR-PPAR-gamma": "activity in the PPAR-gamma receptor assay",
    "SR-ARE": "activity in the antioxidant responsive element assay",
    "SR-ATAD5": "activity in the ATPase Family AAA Domain Containing 5e assay",
}


target_number_mapping = {
    "NR-AR": 0,
    "NR-AR-LBD": 1,
    "NR-AhR": 2,
    "NR-Aromatase": 3,
    "NR-ER": 4,
    "NR-ER-LBD": 5,
    "NR-PPAR-gamma": 6,
    "SR-ARE": 7,
    "SR-ATAD5": 8,
}


def run_experiment(target, num_train_points, random_undersample, num_test_points, seed):
    tox21_tasks, tox21_datasets, transformers = load_tox21(seed=seed, reload=False)
    train_dataset, valid_dataset, test_dataset = tox21_datasets

    X_train, y_train = train_dataset.ids, train_dataset.y[:, target_number_mapping[target]]
    X_test, y_test = test_dataset.ids, test_dataset.y[:, target_number_mapping[target]]

    if num_train_points == "max":
        num_train_points = len(X_train)
    if random_undersample:
        sampler = RandomUnderSampler(random_state=seed)

        X_train, y_train = sampler.fit_resample(X_train.reshape(-1, 1), y_train)

    train_ids = np.random.choice(np.arange(len(X_train)), num_train_points, replace=False)
    test_ids = np.random.choice(np.arange(len(X_test)), num_test_points, replace=False)

    X_train = X_train[train_ids]
    y_train = y_train[train_ids]

    X_test = X_test[test_ids]
    y_test = y_test[test_ids]
    n_epochs = 8

    tuner = Tuner(n_epochs=n_epochs, learning_rate_multiplier=0.02, wandb_sync=False)
    classifier = GPTClassifier(
        target,
        tuner=tuner,
        save_valid_file=True,
        querier_settings={"max_tokens": 10},
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    results = evaluate_classification(y_test, y_pred)

    report = {
        "target": target,
        "num_train_points": num_train_points,
        "num_test_points": num_test_points,
        "random_undersample": random_undersample,
        "seed": seed,
        "y_pred": y_pred,
        "y_test": y_test,
        "n_epochs": n_epochs,
        "short_name": True,
        **results,
    }

    timestr = get_timestr()

    save_pickle(
        f"reports/{timestr}-{target}-{num_train_points}-{random_undersample}-{seed}-{n_epochs}.pkl",
        report,
    )


def get_grid(random_undersample):
    if random_undersample:
        return [10, 50, 100]
    else:
        return [10, 100, 6000, "max"]


if __name__ == "__main__":
    for seed in range(3):
        seed = seed + 54535
        for random_undersample in [True, False][::-1]:
            for target in list(name_mapping.keys())[::-1]:
                for num_train_points in get_grid(random_undersample)[::-1]:
                    try:
                        run_experiment(target, num_train_points, random_undersample, 500, seed)
                        time.sleep(60)
                    except Exception as e:
                        print(e)
                        time.sleep(60)
