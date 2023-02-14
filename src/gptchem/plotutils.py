from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def range_frame(ax, x, y, pad=0.1):
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()

    ax.set_ylim(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min))
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))

    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.spines["bottom"].set_bounds(x_min, x_max)
    ax.spines["left"].set_bounds(y_min, y_max)


def ylabel_top(
    string: str, ax: Optional[plt.Axes] = None, x_pad: float = 0.01, y_pad: float = 0.02
) -> None:
    # Rotate the ylabel (such that you can read it comfortably) and place it
    # above the top ytick. This requires some logic, so it cannot be
    # incorporated in `style`. See
    # <https://stackoverflow.com/a/27919217/353337> on how to get the axes
    # coordinates of the top ytick.
    if ax is None:
        ax = plt.gca()

    yticks_pos = ax.get_yticks()
    coords = np.column_stack([np.zeros_like(yticks_pos), yticks_pos])
    data_to_axis = ax.transData + ax.transAxes.inverted()
    yticks_pos_ax = data_to_axis.transform(coords)[:, 1]
    # filter out the ticks which aren't shown
    tol = 1.0e-5
    yticks_pos_ax = yticks_pos_ax[(-tol < yticks_pos_ax) & (yticks_pos_ax < 1.0 + tol)]
    if len(yticks_pos_ax) > 0:
        pos_y = yticks_pos_ax[-1] + 0.1
    else:
        pos_y = 1.0

    # Get the padding in axes coordinates. The below logic isn't quite correct, so keep
    # an eye on <https://stackoverflow.com/q/67872207/353337> and
    # <https://discourse.matplotlib.org/t/get-ytick-label-distance-in-axis-coordinates/22210>
    # and
    # <https://github.com/matplotlib/matplotlib/issues/20677>
    yticks = ax.yaxis.get_major_ticks()
    if len(yticks) == 0:
        pos_x = 0.0
    else:
        pad_pt = yticks[-1].get_pad()
        # https://stackoverflow.com/a/51213884/353337
        # ticklen_pt = ax.yaxis.majorTicks[0].tick1line.get_markersize()
        # dist_in = (pad_pt + ticklen_pt) / 72.0
        dist_in = pad_pt / 72.0
        # get axes width in inches
        # https://stackoverflow.com/a/19306776/353337
        bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        pos_x = -dist_in / bbox.width

    yl = ax.set_ylabel(string, horizontalalignment="right", multialignment="right")
    # place the label 10% above the top tick
    ax.yaxis.set_label_coords(pos_x - x_pad, pos_y + y_pad)
    yl.set_rotation(0)


def add_identity(axes, *line_args, **line_kwargs):
    (identity,) = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    return axes
