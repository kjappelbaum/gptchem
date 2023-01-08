# -*- coding: utf-8 -*-
"""Constants for the gptchem package."""
import os

import pystow
from scipy.constants import golden

__all__ = ["GPTCHEM_PYSTOW_MODULE", "BASE_OUTDIR"]

GPTCHEM_PYSTOW_MODULE = pystow.module("gptchem")


BASE_OUTDIR = os.getenv("GPTCHEM_OUTDIR", "out")

ONE_COL_WIDTH_INCH = 3.25
TWO_COL_WIDTH_INCH = 7.2

ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden
