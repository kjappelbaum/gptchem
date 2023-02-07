from pathlib import Path

import pandas as pd
from fastcore.all import L
from fastcore.xtras import save_pickle
from loguru import logger

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
NOISE_LEVEL = [0.1, 0.2, 0.5, 1.0, 5.0, 10, 20, 50]
NUM_SAMPLES = 300
GROUPS = ["C#CBr", "C#CC", "C=O", "C#C", "I",  "Br",  "Cl", "F"][::-1]

THRESHOLD = 350

logger.enable('gptchem')

def get_prevalence(df, fragment):
    return df["SMILES"].apply(lambda x: fragment in x).sum() / len(df)



def train_test_evaluate(train_size, noise_level, num_samples, temperatures, group, seed):
    data = get_photoswitch_data()
    prevalence = get_prevalence(data, group)
    data_subset = data.dropna(
        subset=["E isomer pi-pi* wavelength in nm", "SMILES", "Z isomer pi-pi* wavelength in nm"]
    )
    formatter = InverseDesignFormatter(
        representation_column="SMILES",
        property_columns=["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"],
        property_names=["E isomer transition wavelength", "Z isomer transition wavelength"],
        num_digits=0,
    )

    formatter._suffix = f" and {group} as part of the molecule?"
    formatted_train = formatter(data_subset.sample(n=train_size, random_state=seed))
    assert f" and {group} as part of the molecule?" in formatted_train["prompt"].iloc[0]

    data_test = data_subset.copy()
    data_test[
        ["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"]
    ] = noise_original_data(
        data_test[["E isomer pi-pi* wavelength in nm", "Z isomer pi-pi* wavelength in nm"]],
        noise_level=noise_level,
    )

    test_size = min(num_samples, len(data_test))
    logger.info(f"Test size: {test_size}")
    formatted_test = formatter(data_test.sample(test_size))

    assert (
        "prompt" in formatted_test.columns
    ), f"Missing prompt column. Columns: {formatted_test.columns}"

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(formatted_train)
    querier = Querier(tune_res["model_name"], max_tokens=600)
    extractor = InverseExtractor()

    train_smiles = formatted_train["label"]
    valid_smiles =formatted_test["label"]

    all_smiles = train_smiles + valid_smiles
    res_at_temp = []
    for temp in temperatures:
        try:
            logger.debug(f"Temperature: {temp}")
            completions = querier(formatted_test, temperature=temp)
            generated_smiles = extractor(completions)
            logger.debug(f"Generated {len(generated_smiles)} SMILES. Examples: {generated_smiles[:5]}")
            smiles_metrics = evaluate_generated_smiles(generated_smiles, formatted_train["label"])
            logger.debug(f"SMILES metrics: {smiles_metrics}")

            smiles_metrics_all = evaluate_generated_smiles(generated_smiles, all_smiles)
            logger.debug(f"SMILES metrics (all): {smiles_metrics_all}")
            assert len(smiles_metrics["valid_indices"]) <= len(generated_smiles), "Found more valid SMILES than generated"
            expected_e = []
            expected_z = []
            for i, row in formatted_test.iterrows():
                expected_e.append(row["representation"][0])
                expected_z.append(row["representation"][1])
            assert len(expected_e) == len(expected_z) == len(formatted_test)

        
            if len(smiles_metrics["valid_indices"]) > 0:
                expected_e_v = L(expected_e)[smiles_metrics["valid_indices"]]
                expected_z_v = L(expected_z)[smiles_metrics["valid_indices"]]

                fragment_count = sum([group in x for x in smiles_metrics["valid_smiles"] ])
                logger.debug(f"Fragment count: {fragment_count}")
                fragment_fraction = fragment_count / len(smiles_metrics["valid_smiles"])

                assert len(expected_e_v) == len(expected_z_v) == len(smiles_metrics["valid_indices"]), "Length mismatch for valid SMILES"

                constrain_satisfaction = evaluate_photoswitch_smiles_pred(
                    smiles_metrics["valid_smiles"],
                    expected_z_pi_pi_star=expected_z_v,
                    expected_e_pi_pi_star=expected_e_v,
                )
                logger.debug(f"Constrain satisfaction: {constrain_satisfaction}")

                # now, let's do the thing for the novel SMILES only 
                expected_e_novel = L(expected_e)[smiles_metrics["novel_indices"]]
                expected_z_novel = L(expected_z)[smiles_metrics["novel_indices"]]

                assert len(expected_e_novel) == len(expected_z_novel) == len(smiles_metrics["novel_indices"]), "Length mismatch for novel SMILES"

                constrain_satisfaction_novel = evaluate_photoswitch_smiles_pred(
                    smiles_metrics["novel_smiles"],
                    expected_z_pi_pi_star=expected_z_novel,
                    expected_e_pi_pi_star=expected_e_novel,
                )

                logger.debug(f"Constrain satisfaction novel: {constrain_satisfaction_novel}")
            else:
                logger.debug("No valid SMILES found")
                constrain_satisfaction = evaluate_photoswitch_smiles_pred(
                    None, expected_z, expected_e
                )

                constrain_satisfaction_novel = evaluate_photoswitch_smiles_pred(
                    None, expected_z, expected_e
                )
        except Exception as e:
            logger.exception(f"Error evaluating SMILES: {e}")
            constrain_satisfaction = {}
            constrain_satisfaction_novel = {}
            fragment_fraction = None

        res = {
            "completions": completions,
            "generated_smiles": generated_smiles,
            "train_smiles": formatted_train["label"],
            **smiles_metrics,
            'constrain_satisfaction': constrain_satisfaction, 
            "constrain_satisfaction_novel":constrain_satisfaction_novel , 
            "temperature": temp,
            "prevalence": prevalence,
            "fragment_fraction": fragment_fraction,
            "formatted_test": formatted_test,
            "group": group,
        }

        res_at_temp.append(res)

    summary = {
        "train_size": train_size,
        "noise_level": noise_level,
        "num_samples": num_samples,
        "temperatures": temperatures,
        "res_at_temp": res_at_temp,
        "test_size": len(formatted_test),
        "threshold": THRESHOLD,
       # "formatted_test": formatted_test
        **tune_res,
        "smiles_metrics_all": smiles_metrics_all, 
        "group": group
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_trials):
        for noise_level in NOISE_LEVEL:
            for group in GROUPS:
                try:
                    train_test_evaluate(TRAIN_SIZE, noise_level, NUM_SAMPLES, TEMPERATURES, group, seed +456)
                except Exception as e:
                    logger.exception(e)
                    continue
