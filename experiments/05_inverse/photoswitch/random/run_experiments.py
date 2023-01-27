from pathlib import Path

from fastcore.all import L
from fastcore.xtras import save_pickle

from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_generated_smiles, evaluate_photoswitch_smiles_pred
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.generator import noise_original_data
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_trials = 10
TRAIN_SIZE = 92
TEMPERATURES = [0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
NOISE_LEVEL = [0.5, 1.0, 5.0, 10, 20, 50]
NUM_SAMPLES = 100


def train_test_evaluate(train_size, noise_level, num_samples, temperatures, seed):
    data = get_photoswitch_data()

    data_subset = data.dropna(
        subset=["E isomer pi-pi* wavelength in nm", "SMILES", "Z isomer pi-pi* wavelength in nm"]
    )
    formatter = InverseDesignFormatter(
        representation_column="SMILES",
        property_columns=["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"],
        property_names=["E isomer transition wavelength", "Z isomer transition wavelength"],
        num_digits=0,
    )

    formatted_train = formatter(data_subset.sample(n=train_size, random_state=seed))

    data_test = data_subset.copy()
    data_test[
        ["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"]
    ] = noise_original_data(
        data_test[["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"]],
        noise_level=noise_level,
    )

    test_size = min(num_samples, len(data_test))

    formatted_test = formatter(data_test.sample(test_size))

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(formatted_train)
    querier = Querier(tune_res["model_name"], max_tokens=600)
    extractor = InverseExtractor()

    res_at_temp = []
    for temp in temperatures:
        completions = querier(formatted_test, temperature=temp)
        generated_smiles = extractor(completions)
        smiles_metrics = evaluate_generated_smiles(generated_smiles, formatted_train["label"])
        assert len(smiles_metrics["valid_indices"]) <= len(generated_smiles)
        expected_e = []
        expected_z = []
        for i, row in formatted_test.iterrows():
            expected_e.append(row["representation"][0])
            expected_z.append(row["representation"][1])
        assert len(expected_e) == len(expected_z) == len(formatted_test)

        try:
            if len(smiles_metrics["valid_indices"]) > 0:
                expected_e = L(expected_e)[smiles_metrics["valid_indices"]]
                expected_z = L(expected_z)[smiles_metrics["valid_indices"]]

                constrain_satisfaction = evaluate_photoswitch_smiles_pred(
                    smiles_metrics["valid_smiles"],
                    expected_z_pi_pi_star=expected_z,
                    expected_e_pi_pi_star=expected_e,
                )

            else:
                constrain_satisfaction = evaluate_photoswitch_smiles_pred(
                    None, expected_z, expected_e
                )
        except Exception as e:
            print(e)
            constrain_satisfaction = {}

        res = {
            "completions": completions,
            "generated_smiles": generated_smiles,
            "train_smiles": formatted_train["label"],
            **smiles_metrics,
            **constrain_satisfaction,
            "temperature": temp,
        }

        res_at_temp.append(res)

    summary = {
        "train_size": train_size,
        "noise_level": noise_level,
        "num_samples": num_samples,
        "temperatures": temperatures,
        "res_at_temp": res_at_temp,
        "test_size": test_size,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_trials):
        for noise_level in NOISE_LEVEL:
            train_test_evaluate(TRAIN_SIZE, noise_level, NUM_SAMPLES, TEMPERATURES, seed+56545634)
