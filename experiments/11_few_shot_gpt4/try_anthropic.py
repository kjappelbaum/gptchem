from gptchem.extractor import FewShotClassificationExtractor
from gptchem.formatter import FewShotFormatter
from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classification
import pandas as pd 
import numpy as np
import os 
from pathlib import Path 
from fastcore.xtras import save_pickle
import time
from sklearn.model_selection import train_test_split

from langchain.llms import Anthropic

MAX_TEST_SIZE = 200

outdir = 'results_anthropic'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def train_test(representation, train_size, num_classes, seed, temperature):

    data = get_photoswitch_data()
    data = data.dropna(subset=["E isomer pi-pi* wavelength in nm", representation])
    data["binned"] = pd.qcut(
        data["E isomer pi-pi* wavelength in nm"], num_classes, labels=np.arange(num_classes)
    )
    train, test = train_test_split(
        data, train_size=train_size, random_state=seed, stratify=data["binned"]
    )
    formatter = FewShotFormatter(
        train,
        property_name="transition wavelength",
        representation_column=representation,
        label_column="binned",
    )

    if num_classes == 2:
        formatter._PREFIX = "You are a highly intelligent chemisty question answering bot that answers questions about {property}. Answer by simply returning only 0 or 1, no other text is necessary."
    else:
        formatter._PREFIX = "You are a highly intelligent chemisty question answering bot that answers questions about {property}. Answer by simply returning only 0, 1, 2, 3 or 4 no other text is necessary."

    test = test.sample(min(len(test), MAX_TEST_SIZE), random_state=seed)
    formatted = formatter(test)

    llm = Anthropic(anthropic_api_key='sk-PxG6E4m8pVJbFXGutrSAe1WB68sNYz3uI8fB9wzfiA-Z1OeqUyFm5F5yx3k34i4xyNyrmY8WgXnoChH-Rx-_mQ', temperature=temperature)

    extractor = FewShotClassificationExtractor()

    predictions = []
    true = []
    for i, row in formatted.iterrows():
        pred = llm(row['prompt'])

        extracted = extractor.extract(pred.replace(row['representation'], ''))
        predictions.append(extracted)
        true.append(row['label'])

    metrics = evaluate_classification(true, predictions)

    print(f"Finished querying {representation} {train_size} {num_classes} {seed} {temperature} {metrics['accuracy']}")
    summary = {
        'representation': representation,
        'train_size': train_size,
        'num_classes': num_classes,
        'seed': seed,
        'temperature': temperature,
        'metrics': metrics
    }

    timestr = time.strftime("%Y%m%d-%H%M%S")

    filename = f"{timestr}_{representation}_{train_size}_{num_classes}_{seed}_{temperature}.pkl"

    save_path = os.path.join(outdir, filename)
    save_pickle(save_path, summary)


if __name__ == '__main__':
    for seed in range(10):
        for num_classes in [2, 5]:
             for representation in ["SMILES", "name",  "inchi", "selfies" ]:
                for train_size in [10, 70, 20, 50]:
                    for temperature in [0]:#, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]:
                        try:
                            train_test(representation, train_size, num_classes, seed + 674567, temperature)
                        except Exception as e:
                            print(f"Failed {representation} {train_size} {num_classes} {seed} {temperature}")
                            print(e)