from loguru import logger

from gptchem.data import get_doyle_rxn_data

logger.enable("gptchem")
from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.baselines.rxn import train_test_rxn_regressions_baseline
from gptchem.evaluator import get_regression_metrics
from gptchem.extractor import RegressionExtractor
from gptchem.formatter import ReactionRegressionFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner

train_sizes = [10, 20, 50, 100, 200]
num_test_points = 100
use_one_hot = [True, False]
num_repeats = 10


def train_test_model(representation, num_train_points, seed, one_hot):
    data = get_doyle_rxn_data()
    bins = data["yield"] > data["yield"].median()

    train_data, test_data = train_test_split(
        data,
        train_size=num_train_points,
        test_size=min((num_test_points, len(data) - num_train_points)),
        stratify=bins,
        random_state=seed,
    )

    formatter = ReactionRegressionFormatter.from_preset("DreherDoyle", num_digit=0, one_hot=one_hot)

    train_formatted = formatter(train_data)
    test_formatted = formatter(test_data)

    baseline_res = train_test_rxn_regressions_baseline(
        "DreherDoyle",
        train_data=train_data,
        test_data=test_data,
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
        "baseline": baseline_res,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)

    print(
        f"Ran train size {num_train_points} and got MAE {res['mean_absolute_error']}, baseline {baseline_res['rxnfp-rbf']['mean_absolute_error']}"
    )


if __name__ == "__main__":
    for seed in range(num_repeats):
        for oh in use_one_hot:
            for train_size in train_sizes:
                try:
                    train_test_model("DreherDoyle", train_size, seed + 9844, oh)
                except Exception as e:
                    print(e)
