from typing import Iterable

import numpy as np
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, MolToSmiles


def compute_morgan_fingerprints(
    smiles_list: Iterable[str],  # list of SMILEs
    n_bits: int = 2048,  # number of bits in the fingerprint
) -> np.ndarray:
    rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
    rdkit_smiles = [MolToSmiles(mol, isomericSmiles=False) for mol in rdkit_mols]
    rdkit_mols = [MolFromSmiles(smiles) for smiles in rdkit_smiles]
    X = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits) for mol in rdkit_mols]
    X = np.asarray(X)
    return X


def compute_fragprints(smiles_list: Iterable[str]) -> np.ndarray:  # list of SMILEs
    X = compute_morgan_fingerprints(smiles_list)

    fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
    X1 = np.zeros((len(smiles_list), len(fragments)))
    for i in range(len(smiles_list)):
        mol = MolFromSmiles(smiles_list[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        X1[i, :] = features

    X = np.concatenate((X, X1), axis=1)
    return X
