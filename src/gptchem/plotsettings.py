import os
import sys

import matplotlib.pyplot as plt
import matplotx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use(os.path.join(_THIS_DIR, "kevin.mplstyle"))
import numpy as np

from .plotutils import add_identity, range_frame, ylabel_top
