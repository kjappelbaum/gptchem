from gptchem.representation import (
    smiles_to_inchi,
    smiles_to_tucan,
    smiles_to_iupac_name,
    line_reps_from_smiles,
    smiles_to_max_random,
    smiles_augment_df,
)
from gptchem.data import get_photoswitch_data


def test_smiles_to_inchi():
    assert smiles_to_inchi("CCO") == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"


def test_smiles_to_tucan():
    assert smiles_to_tucan("CCO") == "C2O/(1-2)(2-3)"


def test_smiles_to_iupac_name():
    assert smiles_to_iupac_name("CCO") == "ethanol"


def test_smiles_to_max_random():
    assert len(smiles_to_max_random("CCO")) == 4


def test_line_reps_from_smiles():
    res = line_reps_from_smiles("CCO")
    assert res["smiles"] == "CCO"
    assert res["deepsmiles"] == "CCO"
    assert len(res["max_random"]) == 4


def test_smiles_augment_df():
    data = get_photoswitch_data()
    augmented = smiles_augment_df(data, "SMILES", int_aug=10, deduplicate=True)

    assert len(augmented) > len(data)
