from gptchem.data import get_hea_phase_data
from sklearn.model_selection import train_test_split
from gptchem.evaluator import evaluate_classification
from automatminer import MatPipe
from pathlib import Path 
from fastcore.xtras import save_pickle
import time

NUM_REPEATS = 10
LEARNING_CURVE_POINTS = [10, 20, 50, 100, 200]
TEST_SIZE = 250


OURDIR = "out-baseline"
Path(OURDIR).mkdir(exist_ok=True)

def train_test_evaluate(train_size, seed):

    data = get_hea_phase_data()
    data['composition'] = data['Alloy']
    data = data[['composition', 'phase_binary_encoded']]


    pipe = MatPipe.from_preset("express")

    train, test = train_test_split(
        data,
        train_size=train_size,
        test_size=TEST_SIZE,
        stratify=data["phase_binary_encoded"],
        random_state=seed,
    )

    pipe.fit(train, "phase_binary_encoded")

    predictions = pipe.predict(test)

    metrics = evaluate_classification(test['phase_binary_encoded'].values, predictions['phase_binary_encoded predicted'].values)

    summary = {
        **metrics,
        "train_size": train_size,
    }

    # use the timestr to make sure we don't overwrite any files
    run_outdir = Path(OURDIR) / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    run_outdir.mkdir(exist_ok=True)
    save_pickle(run_outdir / "summary.pkl", summary)

    return summary

if __name__ == '__main__':
    for seed in range(NUM_REPEATS):
        for train_size in LEARNING_CURVE_POINTS:
            train_test_evaluate(train_size, seed)