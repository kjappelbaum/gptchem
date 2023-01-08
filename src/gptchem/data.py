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
        )
        .reset_index(drop=True)
    )

def get_moosavi_mof_data() ->  pd.DataFrame:
    """Return the data and features used in [MoosaviDiversity]_.

    You can find the original datasets on `MaterialsCloud archive <https://archive.materialscloud.org/record/2020.67>`_.

    We additionally computed the MOFid [BuciorMOFid]_ for each MOF.
    """