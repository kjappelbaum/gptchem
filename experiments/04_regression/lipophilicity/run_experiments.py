from gptchem.data import get_lipophilicity_data
from loguru import logger

logger.enable("gptchem")
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import RegressionFormatter
from gptchem.tuner import Tuner
from gptchem.querier import Querier
from gptchem.baselines.freesolv import train_test_freesolv_regression_baseline

from gptchem.evaluator import get_regression_metrics

from sklearn.model_selection import train_test_split

from pathlib import Path
from fastcore.xtras import save_pickle

num_training_points = [10, 20, 50, 100, 200, 1000, 5000]  # 1000
num_training_points = [10, 50, 100, 200, 500]  # 1000
representations = ["smiles", "inchi", "selfies", "iupac_name"]
max_num_test_points = 250
num_repeats = 10


def train_test_model(representation, num_train_points, seed):
    data = get_lipophilicity_data()
    bins = (
        data["exp"] > data["exp"].median()
    )

    train_data, test_data = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min((max_num_test_points, len(data) - num_train_points)),
        stratify=bins,
        random_state=seed,
    )

    train_smiles = train_data["smiles"].values
    test_smiles = test_data["smiles"].values

    formatter = RegressionFormatter(
        representation_column=representation,
        property_name="lipophilicity",
        label_column="exp",
    )

    train_formatted = formatter(train_data)
    test_formatted = formatter(test_data)

    gpr_baseline = train_test_freesolv_regression_baseline(
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


    print(
        f"Ran train size {num_train_points} and got MAE {res['mean_absolute_error']}, GPR baseline {gpr_baseline['mean_absolute_error']}"
    )



if __name__ == "__main__":
    for seed in range(num_repeats):
        for representation in representations:
            for num_train_points in num_training_points:
                train_test_model(representation, num_train_points, seed + 3546)
