from pathlib import Path

from fastcore.xtras import save_pickle
from sklearn.model_selection import train_test_split

from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner
from loguru import logger
num_classes = [2, 5]
num_training_points = [10, 20, 50, 100, 200]  # 1000
representations = ["name", "SMILES", "inchi", "selfies"]
max_num_test_points = 100
num_repeats = 10

logger.enable("gptchem")

def train_test_model(num_classes, representation, num_train_points, seed):
    data = get_photoswitch_data()

    data['E isomer pi-pi* wavelength in nm'] = data['E isomer pi-pi* wavelength in nm'].sample(frac=1, random_state=seed)

    formatter = ClassificationFormatter(
        representation_column=representation,
        label_column="E isomer pi-pi* wavelength in nm",
        property_name="transition wavelength",
        num_classes=num_classes,
    )
    formatted = formatter(data)
    num_test_points = min((max_num_test_points, len(formatted) - num_train_points))

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
        f"Ran train size {num_train_points} and got accuracy {gpt_metrics['accuracy']}"
    )

    summary = {
        **gpt_metrics,
        "train_size": num_train_points,
        "num_classes": num_classes,
        "completions": completions,
        "extracted": extracted,
        "representation": representation,
        "num_test_points": num_test_points,
    }

    save_pickle(Path(tune_res["outdir"]) / "summary.pkl", summary)


if __name__ == "__main__":
    for i in range(num_repeats):
        for num_class in num_classes:
            for num_train_points in num_training_points:
                for representation in representations:
                    try:
                        train_test_model(num_class, representation, num_train_points, i + 3436545)
                    except Exception as e:
                        print(e)
