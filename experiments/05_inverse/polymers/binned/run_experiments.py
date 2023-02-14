from pathlib import Path

from fastcore.xtras import save_pickle
from loguru import logger
from more_itertools import pairwise
from sklearn.model_selection import train_test_split

from gptchem.data import get_polymer_data
from gptchem.evaluator import (
    PolymerKLDivBenchmark,
    get_inverse_polymer_metrics,
    polymer_string2performance,
    string_distances,
)
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.generator import noise_original_data
from gptchem.querier import Querier
from gptchem.tuner import Tuner

repeats = 10
num_samples = 100
num_train_points = [100, 300, 1000]

temperatures = [0.0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5]

noise_levels = [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]

logger.enable("gptchem")


def train_test(num_train_points, temperatures, num_sample, noise_level, seed):
    data = get_polymer_data()

    data_test = data.copy()
    data_test[["deltaGmin"]] = noise_original_data(
        data_test[["deltaGmin"]],
        noise_level=noise_level,
    ).sample(num_sample)

    formatter = InverseDesignFormatter(
        representation_column="string",
        property_columns=["deltaGmin"],
        property_names=["adsorption_energy"],
        num_classes=5,
    )

    formatted_train = formatter(data.sample(n=num_train_points, random_state=seed))

    formatted_test = formatter(data_test)

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    tune_res = tuner(formatted_train)

    querier = Querier(tune_res["model_name"], max_tokens=600)
    extractor = InverseExtractor()

    res_at_temp = []
    for temp in temperatures:
        try:
            logger.info(f"Temperature: {temp}")
            completions = querier(formatted_test, temperature=temp)
            generated_smiles = extractor(completions)
            results = get_inverse_polymer_metrics(
                generated_smiles,
                formatted_train,
                formatted_test,
                bins=list(pairwise(formatter.bins)),
            )
            results["temp"] = temp
            res_at_temp.append(results)
        except Exception as e:
            logger.error(e)
            continue

    summary = {"num_train_points": num_train_points, "noise_level": noise_level, "res": res_at_temp}
    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(repeats):
        for train_point in num_train_points:
            for noise_level in noise_levels:
                train_test(train_point, temperatures, num_samples, noise_level, i)
