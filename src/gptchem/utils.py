import os
import time

from .settings import BASE_OUTDIR


def make_outdir(run_name):
    """Make a directory if the current date and time in the format YYYYMMDD_HHMMSS.

    If run_name is specified, append it to the directory name in the format YYYYMMDD_HHMMSS_run_name.
    """
    outdir = os.path.abspath(os.path.join(BASE_OUTDIR, time.strftime("%Y%m%d_%H%M%S")))
    if run_name is not None:
        outdir = f"{outdir}_{run_name}"
    os.makedirs(outdir, exist_ok=True)
    return outdir
