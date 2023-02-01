from loguru import logger

from gptchem.data import get_photoswitch_data

logger.enable("gptchem")
from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.bandgap import train_test_bandgap_regression_baseline
from gptchem.evaluator import get_regression_metrics
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_training_points = [10, 20, 50, 100, 200][::-1]  # 1000
representations = ["name", "SMILES", "inchi", "selfies"][::-1]
max_num_test_points = 100
num_repeats = 10


def train_test_model(representation, num_train_points, seed):
    data = get_photoswitch_data()
    bins = (
        data["E isomer pi-pi* wavelength in nm"] > data["E isomer pi-pi* wavelength in nm"].median()
    )

    train_data, test_data = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min((max_num_test_points, len(data) - num_train_points)),
        stratify=bins,
        random_state=seed,
    )

    train_smiles = train_data["SMILES"].values
    test_smiles = test_data["SMILES"].values

    formatter = RegressionFormatter(
        representation_column=representation,
        label_column="E isomer pi-pi* wavelength in nm",
        property_name="transition wavelength",
    )

    train_formatted = formatter(train_data)
    test_formatted = formatter(test_data)

    gpr_baseline = train_test_bandgap_regression_baseline(
        data, train_smiles=train_smiles, test_smiles=test_smiles, formatter=formatter
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    tune_res = tuner(train_formatted)
    querier = Querier(tune_res["model_name"])
    completions = querier(test_formatted)
    extractor = RegressionExtractor()
    extracted = extractor(completions)

    res = get_regression_metrics(test_formatted["label"].values, extracted)

    summary = {
        "representation": representation,
        "num_train_points": num_train_points,
        **res,
        "gpr_baseline": gpr_baseline,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_repeats):
        for representation in representations:
            for num_train_points in num_training_points:
                train_test_model(representation, num_train_points, seed + 1224)
