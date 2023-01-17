from pathlib import Path

from fastcore.all import L
from fastcore.xtras import save_pickle

from gptchem.data import get_qmug_data
from gptchem.evaluator import evaluate_generated_smiles, evaluate_homo_lumo_gap
from gptchem.extractor import InverseExtractor
from gptchem.formatter import InverseDesignFormatter
from gptchem.generator import noise_original_data
from gptchem.querier import Querier
from gptchem.tuner import Tuner

num_samples = 10
num_train_points = [100]

temperatures = [0.5]

noise_levels = [1.0]


def train_test(num_train_points, temperatures, num_samples, seed):
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
        expected = []
        for i, row in formatted_test.iterrows():
            expected.append(row["representation"][0])

        try:
            if len(smiles_metrics["valid_indices"]) > 0:
                expected = L(expected)[smiles_metrics["valid_indices"]]

                constrain_satisfaction = evaluate_homo_lumo_gap(
                    smiles_metrics["valid_smiles"],
                    expected_gaps=expected,
                )
            else:
                constrain_satisfaction = evaluate_homo_lumo_gap(
                    None,
                    expected_gaps=expected,
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
        }
        res_at_temp.append(res)

    summary = {
        "train_size": num_train_points,
        "noise_level": noise_level,
        "num_samples": num_samples,
        "temperatures": temperatures,
        "res_at_temp": res_at_temp,
        "test_size": test_size,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for seed in range(num_samples):
        for noise_level in noise_levels:
            for num_train_points in num_train_points:
                train_test(num_train_points, temperatures, num_samples, seed)
