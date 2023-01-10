import pandas as pd
import pystow


def get_photoswitch_data() -> pd.DataFrame:
    """Return the photoswitch data as a pandas DataFrame.

    References:
        [GriffithsPhotoSwitches] `Griffiths, K.; Halcovitch, N. R.; Griffin, J. M. Efficient Solid-State Photoswitching of Methoxyazobenzene in a Metal–Organic Framework for Thermal Energy Storage. Chemical Science 2022, 13 (10), 3014–3019. <https://doi.org/10.1039/d2sc00632d>`_
    """
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "photoswitches",
            url="https://www.dropbox.com/s/z5z9z944cc060x9/photoswitches.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).drop_duplicates(subset=['SMILES'])
        .reset_index(drop=True)
    )

def get_polymer_data() -> pd.DataFrame:
    """Return the dataset reported in [JablonkaAL]_."""
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "polymer",
            url="https://www.dropbox.com/s/rpximatxlb8igl9/polymers.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    ) 
    


def get_moosavi_mof_data() ->  pd.DataFrame:
    """Return the data and features used in [MoosaviDiversity]_.

    You can find the original datasets on `MaterialsCloud archive <https://archive.materialscloud.org/record/2020.67>`_.

    We additionally computed the MOFid [BuciorMOFid]_ for each MOF.
    """
    ...


def get_moosavi_cp_data() -> pd.DataFrame:
    """Return the data and features used in [MoosaviCp]_.

    You can find the original datasets on `MaterialsCloud archive <https://doi.org/10.24435/materialscloud:p1-2y>`_.

    We additionally computed the MOFid [BuciorMOFid]_ for each MOF 
    and dropped entries for which we could not compute the MOFid.
    """
    ...


def get_qmug_data() -> pd.DataFrame:
    """Return the data and features used in [QMUG]_.

    We mean-aggregrated the numerical data per SMILES
    and additionally computed SELFIES and INChI.
    """
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "qmug",
            url="https://www.dropbox.com/s/6pk0ohy5agqwe3q/qmugs.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    ) 


def get_hea_phase_data() -> pd.DataFrame:
    """Return the dataset reported in [Pei]_."""
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "hea",
            url="https://www.dropbox.com/s/4edwffuajclxa5h/hea_phase.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    ) 