from gptchem.evaluator import evaluate_classification
from gptchem.querier import Querier
from gptchem.tuner import Tuner 
from gptchem.extractor import ClassificationExtractor
from gptchem.data import get_opv_data
from gptchem.formatter import ClassificationFormatter
from sklearn.model_selection import train_test_split

from fastcore.xtras import save_pickle

from pathlib import Path 
from gptchem.baselines.opv import train_test_opv_classification_baseline

num_classes = [2, 5]
num_training_points = [10, 50, 100, 200, 500] 
representations = ['SMILES', 'SELFIES', 'InChI']
num_test_points = 250
num_repeats = 10


def train_test_model(num_classes, representation, num_train_points, seed):
    data = get_opv_data()
    formatter = ClassificationFormatter(
    representation_column=representation,
    property_name="PCE",
    label_column="PCE_ave(%)",
    num_classes=num_classes
)
    formatted = formatter(data)

    opv_baseline= train_test_opv_classification_baseline(
        data,
        train_size=num_train_points,
        test_size=num_test_points,
        formatter=formatter,
        seed = seed,
    )

    train, test = train_test_split(
        formatted,
        train_size=num_train_points,
        test_size=num_test_points,
        stratify=formatted["label"],
        random_state=seed,
    )   


    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)
    querier = Querier.from_preset(tune_res["model_name"])
    completions = querier(test, logprobs=num_classes)
    extractor = ClassificationExtractor()
    extracted = extractor(completions)

    gpt_metrics = evaluate_classification(test["label"], extracted)

    print(
        f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}, XGB baseline {opv_baseline['xgb']['accuracy']}, and TabPFN baseline {opv_baseline['tabpfn']['accuracy']}"
    )

    summary = {
        "num_classes": num_classes,
        "representation": representation,
        "num_train_points": num_train_points,
        **gpt_metrics,
        **opv_baseline,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_classes in num_classes:
            for num_train_point in num_training_points:
                for representation in representations:
                    try:
                        train_test_model(num_classes, representation, num_train_point, i + 456)
                    except Exception as e:
                        print(e)
