from pathlib import Path

import pandas as pd
from fastcore.all import L
from fastcore.xtras import save_pickle
from loguru import logger

from gptchem.data import get_qmug_data
from gptchem.evaluator import evaluate_generated_smiles, evaluate_photoswitch_smiles_pred
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.generator import noise_original_data
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_trials = 10
TRAIN_SIZE = 92
TEMPERATURES = [0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
NOISE_LEVEL = [0.1,  0.15, 0.5, 0.2,  0.7,  1.0, 5.0, 10, 20, 50]
NUM_SAMPLES = 300
GROUPS = [ "I",  "Br",  "Cl", "F", "C#CC", "C#CBr","C=O", "C#C"]

THRESHOLD = 350

logger.enable('gptchem')

def get_prevalence(df, fragment):
    return df["SMILES"].apply(lambda x: fragment in x).sum() / len(df)



def train_test_evaluate(train_size, noise_level, num_samples, temperatures, group, seed):
    data = get_qmug_data()
    
    
    formatter = InverseDesignFormatter(
        representation_column="SMILES",
        property_columns=["GFN2_HOMO_LUMO_GAP_mean_ev"],
        property_names=["bandgap"],
        num_digits=2,
    )

    formatter._suffix = f" and {group} as part of the molecule?"

    train_data = data
    prevalence = get_prevalence(train_data, group)


    formatted_train = formatter(train_data)
    assert f" and {group} as part of the molecule?" in formatted_train["prompt"].iloc[0]

    data_test = data

    test_size = min(num_samples, len(data_test))
    logger.info(f"Test size: {test_size}")
    formatted_test = formatter(data_test.sample(test_size))

    assert (
        "prompt" in formatted_test.columns
    ), f"Missing prompt column. Columns: {formatted_test.columns}"

    querier = Querier("ada:ft-lsmoepfl-2023-02-09-13-32-32", max_tokens=600)
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
            expected = []
            for i, row in formatted_test.iterrows():
                expected.append(row["representation"][0])
 
            assert len(expected) == len(formatted_test)
            fragment_count = sum([group in x for x in smiles_metrics["valid_smiles"] ])
            logger.debug(f"Fragment count: {fragment_count}")
            fragment_fraction = fragment_count / len(smiles_metrics["valid_smiles"])
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
            "expected": expected,
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
        "smiles_metrics_all": smiles_metrics_all, 
        "group": group
    }

    save_pickle(Path("out2") / Path("summary.pkl"), summary)


if __name__ == "__main__":
    for seed in range(num_trials):
        for noise_level in NOISE_LEVEL:
            for group in GROUPS:
                try:
                    train_test_evaluate(TRAIN_SIZE, noise_level, NUM_SAMPLES, TEMPERATURES, group, seed+34)
                except Exception as e:
                    logger.exception(e)
                    continue
