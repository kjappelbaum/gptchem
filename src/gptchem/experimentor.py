from sklearn.model_selection import train_test_split

from gptchem.evaluator import evaluate_classification
from gptchem.extractor import ClassificationExtractor
from gptchem.formatter import ClassificationFormatter
from gptchem.querier import Querier
from gptchem.tuner import Tuner


def run_classification_experiment(
    df,
    train_size,
    test_size,
    representation_column,
    label_column,
    property_name,
    num_classes,
    qcut,
    basemodel,
    n_epochs,
    learning_rate_multiplier,
    seed,
):

    formatter = ClassificationFormatter(
        representation_column=representation_column,
        label_column=label_column,
        property_name=property_name,
        num_classes=num_classes,
        qcut=qcut,
    )

    formatted = formatter(df)

    train, test = train_test_split(
        formatted,
        train_size=train_size,
        test_size=test_size,
        stratify=formatted["label"],
        random_state=seed,
    )

    tuner = Tuner(
        base_model=basemodel,
        n_epochs=n_epochs,
        learning_rate_multiplier=learning_rate_multiplier,
        wandb_sync=False,
    )

    tune_summary = tuner(train)

    assert isinstance(tune_summary["model_name"], str)

    querier = Querier.from_preset(tune_summary["model_name"], preset="classification")

    completions = querier(test, logprobs=2)

    extractor = ClassificationExtractor()

    extracted = extractor(completions)

    res = evaluate_classification(test["label"], extracted)

    summary = {
        **tune_summary,
        **res,
        "completions": completions,
        "train_size": train_size,
        "test_size": test_size,
    }
    return summary
