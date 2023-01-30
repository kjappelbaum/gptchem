from gptchem.data import get_balasz_data
from gptchem.tuner import Tuner
from gptchem.extractor import ClassificationExtractor
from gptchem.querier import Querier

from loguru import logger
from sklearn.model_selection import train_test_split
from gptchem.baselines.balasz import train_test_mof_synthesizability_baseline
from gptchem.evaluator import evaluate_classification
from gptchem.formatter import MOFSynthesisRecommenderFormatter

from pathlib import Path
from fastcore.xtras import save_pickle

train_sizes = [10, 20, 50, 100, 200, 400]
repeats = 10
threshold = 0.6

logger.enable("gptchem")


def train_test(train_size, seed):
    data = get_balasz_data()
    data["success"] = data["score"] > threshold
    data["success"] = data["success"].astype(int)
    dois = data["reported"].unique()
    train_dois, test_dois = train_test_split(dois, train_size=train_size, random_state=seed)

    train = data[data["reported"].isin(train_dois)]
    test = data[data["reported"].isin(test_dois)]

    baseline_res = train_test_mof_synthesizability_baseline(train, test, "success")

    formatter = MOFSynthesisRecommenderFormatter(score_column="success")
    train_formatted = formatter(train)
    test_formatted = formatter(test)

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train_formatted)

    querier = Querier(tune_res["model_name"])
    completions = querier(test_formatted)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_res = evaluate_classification(extracted, test_formatted)

    res = {
        **baseline_res,
        **gpt_res,
        **tune_res,
        "train_size": train_size,
        "threshold": threshold,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", res)

    return res


if __name__ == "__main__":
    for i in range(repeats):
        for train_size in train_sizes:
            try:
                train_test(train_size, i)
            except Exception as e:
                logger.exception(e)
                continue
