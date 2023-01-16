import os
import sys
import time
from typing import List

from loguru import logger

from .settings import BASE_OUTDIR

__all__ = ["enable_logging", "make_outdir"]


def make_outdir(run_name):
    """Make a directory if the current date and time in the format YYYYMMDD_HHMMSS.

    If run_name is specified, append it to the directory name in the format YYYYMMDD_HHMMSS_run_name.
    """
    outdir = os.path.abspath(os.path.join(BASE_OUTDIR, time.strftime("%Y%m%d_%H%M%S")))
    if run_name is not None:
        outdir = f"{outdir}_{run_name}"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def enable_logging() -> List[int]:
    """Set up the gptchem logging with sane defaults."""
    logger.enable("gptchem")

    config = dict(
        handlers=[
            dict(
                sink=sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
                " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
                " <red>|</> <lvl>{message}</>",
                level="INFO",
            ),
            dict(
                sink=sys.stderr,
                format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}</>",
                level="WARNING",
            ),
        ]
    )
    return logger.configure(**config)
