from gptchem.data import get_photoswitch_data

from gptchem.evaluator import get_regression_metrics
from gptchem.tuner import Tuner 
from sklearn.model_selection import train_test_split
from gptchem.formatter import RegressionFormatter
from gptchem.extractor import RegressionExtractor
from gptchem.querier import Querier

from pathlib import Path 
from fastcore.xtras import save_pickle 

train_points = [10, 20, 50, 100, 200]
max_test_points = 100
repeats = 10 

def train_evaluate(num_train_points, seed):
    data = get_photoswitch_data()

    data = data.dropna(subset=["E isomer pi-pi* wavelength in nm", "SMILES"])

    formatter = RegressionFormatter(
        representation_column="SMILES",
        property_name="transition wavelength",
        label_column="E isomer pi-pi* wavelength in nm",
        num_digits=0)

    formatted = formatter(data)
    binned = data['E isomer pi-pi* wavelength in nm'] > data["E isomer pi-pi* wavelength in nm"].median()

    train, test = train_test_split(
        formatted,
        train_size=num_train_points,
        test_size=min(max_test_points, len(data) - num_train_points),
        stratify=binned,
        random_state=seed,
    )

    tuner = Tuner(n_epochs=8, learning_rate_multiplier=0.02, wandb_sync=False)
    tune_res = tuner(train)

    querier = Querier(tune_res["model"], max_tokens=10)
    completions = querier(test)
    extractor = RegressionExtractor()

    predictions = extractor(completions)

    metrics = get_regression_metrics(
        predictions, test_data["E isomer pi-pi* wavelength in nm"]
    )
    
    
    return metrics