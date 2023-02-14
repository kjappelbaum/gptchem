from pathlib import Path

from fastcore.all import L
from fastcore.xtras import save_pickle
from loguru import logger

from gptchem.data import get_qmug_data
from gptchem.evaluator import evaluate_generated_smiles, evaluate_homo_lumo_gap
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.generator import noise_original_data
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_samples = 100
num_train_points = [100, 300, 1000]

temperatures = [0.0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5]

noise_levels = [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]

logger.enable("gptchem")


def train_test(num_train_points, temperatures, num_samples, noise_level, seed):
    data = get_qmug_data()
    formatter = InverseDesignFormatter(
        representation_column="SMILES",
        property_columns=["GFN2_HOMO_LUMO_GAP_mean_ev"],
        property_names=["bandgap"],
        num_digits=2,
    )

    formatted_train = formatter(data.sample(n=num_train_points, random_state=seed))

    data_test = data.copy()
    data_test[["GFN2_HOMO_LUMO_GAP_mean_ev"]] = noise_original_data(
        data_test[["GFN2_HOMO_LUMO_GAP_mean_ev"]],
        noise_level=noise_level,
    )

    test_size = min(num_samples, len(data_test))
    logger.info(f"Test size: {test_size}")
    formatted_test = formatter(data_test.sample(test_size))

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(formatted_train)
    querier = Querier(tune_res["model_name"], max_tokens=600)
    extractor = InverseExtractor()
    expected = []
    for i, row in formatted_test.iterrows():
        expected.append(row["representation"][0])
    logger.debug(f"Expecting gaps such as {expected[:2]}")
    res_at_temp = []
    for temp in temperatures:
        try:
            logger.info(f"Temperature: {temp}")
            completions = querier(formatted_test, temperature=temp)
            generated_smiles = extractor(completions)
            logger.info(f"Extracted. Evaluating generated SMILES...")
            logger.info(f"generated examples {generated_smiles[:2]}")
            smiles_metrics = evaluate_generated_smiles(generated_smiles, formatted_train["label"])
            smiles_metrics_all = evaluate_generated_smiles(generated_smiles, data["SMILES"])

            logger.debug(f"SMILES metrics (all): {smiles_metrics_all}")
            assert len(smiles_metrics["valid_indices"]) <= len(
                generated_smiles
            ), "Found more valid SMILES than generated"

            logger.info(f"Evaluating constraint satisfaction...")
            try:
                if len(smiles_metrics["valid_indices"]) > 0:
                    expected_v = L(expected)[smiles_metrics["valid_indices"]]
                    expected_e = L(expected)[smiles_metrics["novel_indices"]]

                    constrain_satisfaction_valid = evaluate_homo_lumo_gap(
                        smiles_metrics["valid_smiles"],
                        expected_gaps=expected_v,
                    )
                    logger.debug(f"Constrain satisfaction: {constrain_satisfaction_valid}")

                    constrain_satisfaction_novel = evaluate_homo_lumo_gap(
                        smiles_metrics["novel_smiles"],
                        expected_gaps=expected_e,
                    )
                else:
                    constrain_satisfaction_valid = evaluate_homo_lumo_gap(
                        None,
                        expected_gaps=expected,
                    )
                    constrain_satisfaction_novel = evaluate_homo_lumo_gap(
                        None,
                        expected_gaps=expected,
                    )

            except Exception as e:
                logger.exception(f"{e}")
                constrain_satisfaction_valid, constrain_satisfaction_novel = {}, {}

            res = {
                "completions": completions,
                "generated_smiles": generated_smiles,
                "train_smiles": formatted_train["label"],
                **smiles_metrics,
                "smiles_metrics_all": smiles_metrics_all,
                "constrain_satisfaction": constrain_satisfaction_valid,
                "constrain_satisfaction_novel": constrain_satisfaction_novel,
                "temperature": temp,
            }
            res_at_temp.append(res)

        except Exception as e:
            logger.exception(f"{e}")

    summary = {
        "train_size": num_train_points,
        "noise_level": noise_level,
        "num_samples": num_samples,
        "temperatures": temperatures,
        "res_at_temp": res_at_temp,
        "test_size": len(formatted_test),
        "expected": expected,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_samples):
        for noise_level in noise_levels:
            for num_train_point in num_train_points:
                # try:
                train_test(num_train_point, temperatures, num_samples, noise_level, seed)
                # except Exception as e:
                #    logger.exception(f"{e}")
