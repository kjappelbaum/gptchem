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


def get_moosavi_cv_data() -> pd.DataFrame:
    """Return the gravimetric heat capacity used in [MoosaviCp]_.

    You can find the original datasets on `MaterialsCloud archive <https://doi.org/10.24435/materialscloud:p1-2y>`_.

    We additionally computed the MOFid [BuciorMOFid]_ for each MOF 
    and dropped entries for which we could not compute the MOFid.
    """
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "cv",
            url="https://www.dropbox.com/s/lncrftmdcgn1zdh/cv.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).drop_duplicates(subset='mofid').reset_index(drop=True)
    )



def get_moosavi_pcv_data() -> pd.DataFrame:
    """Return the site-projected heat capacity and features used in [MoosaviCp]_.

    You can find the original datasets on `MaterialsCloud archive <https://doi.org/10.24435/materialscloud:p1-2y>`_.

    We additionally computed the MOFid [BuciorMOFid]_ for each MOF 
    and dropped entries for which we could not compute the MOFid.
    """
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "pcv",
            url="https://www.dropbox.com/s/r4fub4i9nadt1kc/pcv.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )



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

def get_opv_data() -> pd.DataFrame:
    """Return the dataset reported in [NagasawaOPV]_""" 
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "opv",
            url="https://www.dropbox.com/s/a45eu1xw0zkyrmc/opv.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )

def get_esol_data() -> pd.DataFrame:
    """Return the dataset reported in [ESOL]_""" 
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "esol",
            url="https://www.dropbox.com/s/3cchwv5hvxm9mdw/esol.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )


def get_solubility_test_data() -> pd.DataFrame:
    """Return the dataset reported in [soltest]_""" 
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "solubility",
            url="https://www.dropbox.com/s/44rstzg32oqhrlm/solubility_test_set.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )


def get_doyle_rxn_data() -> pd.DataFrame:
    """Return the reaction dataset reported in [Doyle]_"""
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "doyle_rxn",
            url="https://www.dropbox.com/s/gjxatqagwh3cwb6/dreher_doyle.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )

def get_suzuki_rxn_data() -> pd.DataFrame:
    """Return the reaction dataset reported in [Doyle]_"""
    return (
        pystow.module("gptchem")
        .ensure_csv(
            "suzuki_rxn",
            url="https://www.dropbox.com/s/0uv38jgrj2k33u7/suzuki_dreher.csv?dl=1",
            read_csv_kwargs=dict(sep=","),
        ).reset_index(drop=True)
    )