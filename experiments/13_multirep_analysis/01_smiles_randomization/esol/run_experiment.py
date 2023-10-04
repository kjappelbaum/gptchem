from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.data import get_esol_data, get_solubility_test_data
from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from gptchem.representation import smiles_augment_df
import pandas as pd
import numpy as np

num_classes = [2, 5][::-1]
num_training_points = [10, 20, 50, 100, 200, 500]
max_num_test_points = 100
num_repeats = 10


def train_test_model(num_classes, num_augmentation_rounds, deduplicate, num_train_points, seed):
    data = get_esol_data()
    data = data.dropna(subset=["measured log(solubility:mol/L)"])
    train = data.sample(num_train_points)
    formatter = ClassificationFormatter(
        representation_column="SMILES",
        label_column="measured log(solubility:mol/L)",
        property_name="transition wavelength",
        num_classes=num_classes,
    )
    test = get_solubility_test_data()

    if num_augmentation_rounds > 0:
        train = smiles_augment_df(
            train,
            smiles_col="SMILES",
            int_aug=num_augmentation_rounds,
            deduplicate=deduplicate,
            include_canonical=True,
        )
    train = formatter(train)
    test = formatter(test)

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test, logprobs=num_classes)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test["label"], extracted)

    print(f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}")

    summary = {
        **gpt_metrics,
        "train_size": num_train_points,
        "num_classes": num_classes,
        "completions": completions,
        "representation": "SMILES",
        "augmentation_rounds": num_augmentation_rounds,
        "deduplicate": deduplicate,
        "include_canonical": True,
        "train_size_aug": len(train),
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_class in num_classes:
            for num_train_points in num_training_points:
                for num_augment_round in [0, 5, 10, 20][::-1]:
                    if num_augment_round > 0:
                        for deduplicate in [True, False][::-1]:
                            try:
                                train_test_model(
                                    num_classes=num_class,
                                    num_augmentation_rounds=num_augment_round,
                                    deduplicate=deduplicate,
                                    num_train_points=num_train_points,
                                    seed=i + 7456,
                                )
                            except Exception as e:
                                print(e)
                    else:
                        try:
                            train_test_model(
                                num_classes=num_class,
                                num_augmentation_rounds=num_augment_round,
                                deduplicate=False,
                                num_train_points=num_train_points,
                                seed=i + 7456,
                            )
                        except Exception as e:
                            print(e)
