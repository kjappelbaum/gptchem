from pathlib import Path

from fastcore.xtras import save_pickle
from loguru import logger
from sklearn.model_selection import train_test_split

from gptchem.data import get_mof_solvent_data
from gptchem.extractor import SolventExtractor
from gptchem.formatter import MOFSolventRecommenderFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

logger.enable("gptchem")


num_train_size = [10]  # [10, 20, 50, 100, 200]
max_test_size = 100
num_trials = 1


def train_test(num_train_size, seed):
    data = get_mof_solvent_data()
    formatter = MOFSolventRecommenderFormatter(
        linker_columns=["linker_1", "linker_2"],
        node_columns=["core_All_Metals"],
        counter_ion_columns=["counterions1"],
        solvent_columns=["solvent1", "solvent2", "solvent3", "solvent4", "solvent5"],
        solvent_mol_ratio_columns=[
            "sol_molratio1",
            "sol_molratio2",
            "sol_molratio3",
            "sol_molratio4",
            "sol_molratio5",
        ],
    )

    formatted = formatter(data)
    train, test = train_test_split(
        formatted, train_size=num_train_size, test_size=max_test_size, random_state=seed
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)
    querier = Querier(tune_res["model_name"], max_tokens=600)
    completions = querier(test)

    extractor = SolventExtractor()
    extracted = extractor(completions)

    res = {
        "train_size": num_train_size,
        "test_size": len(test),
        "train": train,
        "test": test,
        "completions": completions,
        "extracted": extracted,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", res)


if __name__ == "__main__":
    for seed in range(num_trials):
        for train_size in num_train_size:
            train_test(train_size, seed)
