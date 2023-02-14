from loguru import logger

from gptchem.data import get_esol_data, get_solubility_test_data

logger.enable("gptchem")
from pathlib import Path

from fastcore.xtras import save_pickle

from gptchem.baselines.solubility import train_test_solubility_regression_baseline
from gptchem.evaluator import get_regression_metrics
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from gptchem.utils import make_outdir

num_training_points = [10, 20, 50, 100, 200, 500][::-1]
representations = ["SMILES", "SELFIES", "InChI"][::-1]
num_repeats = 10


def train_test_model(representation, num_train_points, seed):
    train_data = get_esol_data()
    test_data = get_solubility_test_data()

    formatter = RegressionFormatter(
        representation_column=representation,
        property_name="solubility",
        label_column="measured log(solubility:mol/L)",
    )

    train_formatted = formatter(train_data)
    test_formatted = formatter(test_data)

    gpr_baseline = train_test_solubility_regression_baseline(
        train_data,
        test_data,
    )

    # tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    # tune_res = tuner(train_formatted)
    # querier = Querier(tune_res["model_name"])
    # completions = querier(test_formatted)
    # extractor = RegressionExtractor()
    # extracted = extractor(completions)

    # res = get_regression_metrics(test_formatted["label"].values, extracted)

    summary = {
        "representation": representation,
        "num_train_points": num_train_points,
        #  **res,
        "baseline": gpr_baseline,
    }
    outdir = make_outdir("")
    save_pickle(Path(outdir) / "summary.pkl", summary)

    print(f"Ran train size {num_train_points} ")


if __name__ == "__main__":
    for seed in range(num_repeats):
        for representation in representations:
            for num_train_points in num_training_points:
                try:
                    train_test_model(representation, num_train_points, seed + 4567)
                except Exception as e:
                    print(e)
                    continue
