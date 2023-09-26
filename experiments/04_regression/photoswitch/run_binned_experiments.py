from loguru import logger

from gptchem.data import get_photoswitch_data

logger.enable("gptchem")
import time
from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.bandgap import train_test_bandgap_regression_baseline
from gptchem.evaluator import get_regression_metrics
from gptchem.gpt_regressor import BinnedGPTRegressor
from gptchem.tuner import Tuner

num_training_points = [10, 50, 100, 200]  # 1000
desired_accuracies = [5, 10, 20, 30, 50]
representations = ["name", "SMILES", "inchi", "selfies"]
max_num_test_points = 100
num_repeats = 5


def train_test_model(representation, desired_accuracy, num_train_points, seed):
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

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)

    regressor = BinnedGPTRegressor(
        property_name="transition wavelength",
        tuner=tuner,
        desired_accuracy=desired_accuracy,
        equal_bin_sizes=True,
    )

    regressor.fit(
        train_data[representation].values, train_data["E isomer pi-pi* wavelength in nm"].values
    )

    predictions = regressor.predict(test_data[representation].values)

    res = get_regression_metrics(test_data["label"].values, predictions)

    summary = {
        "representation": representation,
        "num_train_points": num_train_points,
        **res,
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_pickle("binned_regression_res" / f"{timestr}_summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_repeats):
        for representation in representations:
            for num_train_points in num_training_points:
                for desired_accuracy in desired_accuracies:
                    try:
                        train_test_model(
                            representation, desired_accuracy, num_train_points, seed + 1224
                        )
                    except Exception as e:
                        logger.exception(e)
