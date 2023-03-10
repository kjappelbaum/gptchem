from pathlib import Path

import fire
import pandas as pd

from gptchem.evaluator import get_homo_lump_gaps


def get_gaps(file):
    with open(file, "r") as handle:
        smiles = handle.readlines()
    gaps = get_homo_lump_gaps(smiles, max_parallel=45)
    df = pd.DataFrame({"smiles": smiles, "gaps": gaps})
    df.to_csv(Path(file).with_suffix(".csv"))


if __name__ == "__main__":
    fire.Fire(get_gaps)
