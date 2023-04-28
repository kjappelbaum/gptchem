from rdkit import Chem
import selfies
import deepsmiles
from tucan.io import graph_from_molfile_text
from tucan.canonicalization import canonicalize_molecule
from tucan.serialization import serialize_molecule

from io import StringIO
import requests
import pubchempy as pcp
import time
import pandas as pd


def augment_smiles(smiles, int_aug=50, deduplicate=True):
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        if int_aug > 0:
            augmented = [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(int_aug)
            ]
            if deduplicate:
                augmented = list(set(augmented))
            return augmented
        else:
            raise ValueError("int_aug must be greater than zero.")


def smiles_to_max_random(smiles, max_duplication=20):
    """
    Returns estimated maximum number of random SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smi_unique = []
    counter = 0
    while counter < max_duplication:
        rand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        if rand not in smi_unique:
            smi_unique.append(rand)
            counter = 0
        else:
            counter += 1
    return smi_unique


def smiles_to_selfies(smiles):
    """
    Takes a SMILES and return the selfies encoding.
    """

    return [selfies.encoder(smiles)]


def smiles_to_deepsmiles(smiles):
    """
    Takes a SMILES and return the DeepSMILES encoding.
    """
    converter = deepsmiles.Converter(rings=True, branches=True)
    return converter.encode(smiles)


def smiles_to_canoncial(smiles):
    """
    Takes a SMILES and return the canoncial SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def smiles_to_inchi(smiles):
    """
    Takes a SMILES and return the InChI.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol)


def smiles_to_tucan(smiles: str):
    """
    Takes a SMILES and return the Tucan encoding.
    For this, create a molfile as StringIO, read it with graph_from_file,
    canonicalize it and serialize it.
    """
    molfile = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))
    mol = graph_from_molfile_text(molfile)
    mol = canonicalize_molecule(mol)
    return serialize_molecule(mol)


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def smiles_to_iupac_name(smiles: str):
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.
    """
    try:
        time.sleep(0.001)
        rep = "iupac_name"
        url = CACTUS.format(smiles, rep)
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        name = response.text
        if "html" in name:
            return None
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "smiles")
            return compound[0].iupac_name
        except Exception:
            return None


def _try_except_none(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def line_reps_from_smiles(smiles: str):
    """
    Takes a SMILES and returns a dictionary with the different representations.
    Use None if some representation cannot be computed.
    """
    representations = {
        "smiles": smiles,
        "selfies": _try_except_none(smiles_to_selfies, smiles),
        "deepsmiles": _try_except_none(smiles_to_deepsmiles, smiles),
        "canonical": _try_except_none(smiles_to_canoncial, smiles),
        "inchi": _try_except_none(smiles_to_inchi, smiles),
        "tucan": _try_except_none(smiles_to_tucan, smiles),
        "iupac_name": _try_except_none(smiles_to_iupac_name, smiles),
        "max_random": _try_except_none(smiles_to_max_random, smiles),
    }
    return representations


def smiles_augment_df(df, smiles_col, int_aug=50, deduplicate=True, include_canonical=True):
    """
    Takes a dataframe with a column of SMILES and returns a dataframe with the augmented SMILES.
    """
    new_rows = []
    for _, row in df.iterrows():
        smiles = []
        if include_canonical:
            smiles.append(smiles_to_canoncial(row[smiles_col]))

        smiles.extend(augment_smiles(row[smiles_col], int_aug=int_aug, deduplicate=False))
        if deduplicate:
            smiles = list(set(smiles))

        for entry in smiles:
            new_row = row.copy()
            new_row[smiles_col] = entry
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)
