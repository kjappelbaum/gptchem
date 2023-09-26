import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastcore.xtras import save_pickle
from loguru import logger
from sklearn.model_selection import train_test_split

from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import FewShotClassificationExtractor
from gptchem.formatter import FewShotFormatter
from gptchem.querier import Querier

models = [
    "gpt-4"
]

representations = ["name", "SMILES", "inchi", "selfies"]

train_sizes = [5, 10, 50, 100]

num_repeats = 5

num_classes = [2, 5]

max_test_size = 50

logger.enable("gptchem")


def train_test(train_size, representation, model, num_classes, seed):
    data = get_photoswitch_data()
    data = data.dropna(subset=["E isomer pi-pi* wavelength in nm"])
    data["binned"] = pd.qcut(
        data["E isomer pi-pi* wavelength in nm"], num_classes, labels=np.arange(num_classes)
    )
    train, test = train_test_split(
        data, train_size=train_size, random_state=seed, stratify=data["binned"]
    )

    formatter = FewShotFormatter(
        train,
        property_name="transition wavelength",
        representation_column=representation,
        label_column="binned",
    )

    test = test.sample(min(len(test), max_test_size), random_state=seed)
    formatted = formatter(test)

    querier = Querier(model, max_tokens=100)
    completions = querier(formatted)

    logger.info(f"Finished querying {model} {representation} {train_size} {num_classes} {seed}")
    logger.info(f"Completion examples: {completions['choices'][:5]}")
    extractor = FewShotClassificationExtractor()
    res = extractor(completions)

    metrics = evaluate_classification(test["binned"], res)

    summary = {
        "model": model,
        "representation": representation,
        "train_size": train_size,
        "num_classes": num_classes,
        "seed": seed,
        "completions": completions,
        "res": res,
        "metrics": metrics,
    }

    logger.info(f"Finished {model} {representation} {train_size} {num_classes} {seed}")
    logger.info(metrics)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model}_{representation}_{train_size}_{num_classes}_{seed}.pkl"
    outdir = Path(os.getcwd()) / "out" / timestr
    outdir.mkdir(parents=True, exist_ok=True)

    save_path = outdir / filename
    save_pickle(save_path, summary)
    formatted.to_csv(save_path.with_suffix(".csv"))


if __name__ == "__main__":
    for seed in range(num_repeats):
        for num_class in num_classes:
            for representation in representations:
                for train_size in train_sizes:
                    for model in models:
                        try:
                            train_test(train_size, representation, model, num_class, seed)
                        except Exception as e:
                            logger.exception(e)
                            pass
