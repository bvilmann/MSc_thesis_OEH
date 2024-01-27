# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 07:46:08 2023

@author: bvilm
"""
# %% ======================== PACKAGES ========================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# %% ======================== CUSTOM SETTINGS ========================
plt.rcParams.update({'lines.markeredgewidth': 1})
plt.rcParams.update({'font.size': 12})
# plt.rcParams['text.usetex'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath} \usepackage{amssymb}"
prop_cycle = plt.rcParams['axes.prop_cycle']
clrs = prop_cycle.by_key()['color']

# %% ======================== CUSTOM PLOT FUNCTIONS ========================
"""
:: Implemented functions ::
    insert_axis():

    update_spine_and_ticks():

    grouped_bar():


"""


def insert_axis(ax, xy_coords: tuple, axin_pos: tuple,
                box_kwargs=dict(color='grey', ls='-'),
                box: bool = True,
                grid: bool = True,
                arrow_pos: tuple = None,
                arrowprops: dict = dict(arrowstyle='->', color='grey', alpha=0.75, ls='-'),
                arrow_kwargs: dict = dict(xycoords='data', textcoords='data'),
                lw = 0.75,
                ):
    xpos, ypos, width, height = axin_pos
    x1, x2, y1, y2 = xy_coords

    axin = ax.inset_axes([xpos, ypos, width, height])

    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)

    # Options
    if arrow_pos is not None:
        x1a, x2a, y1a, y2a = arrow_pos
        ax.annotate('', xy=(x1a, y1a),
                    xytext=(x2a, y2a),
                    arrowprops=arrowprops,
                    **arrow_kwargs
                    )

    if grid:
        axin.grid()
    if box:
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=lw, **box_kwargs)

    return axin, ax


def update_spine_and_ticks(ax,
                           spines='all',
                           color=None,
                           linewidth=None, lw=None,
                           xlabel_color='black', ylabel_color='black',
                           tick_params: dict = dict(colors='black', which='both')):
    if spines == 'all':
        if color is not None:
            ax.spines[:].set_color(color)
        if linewidth is not None:
            ax.spines[:].set_linewidth(linewidth)
        if lw is not None:
            ax.spines[:].set_linewidth(lw)
    else:
        if color is not None:
            ax.spines[spines].set_color(color)
        if linewidth is not None:
            ax.spines[spines].set_linewidth(linewidth)
        if lw is not None:
            ax.spines[spines].set_linewidth(lw)

    ax.xaxis.label.set_color(xlabel_color)
    ax.yaxis.label.set_color(ylabel_color)
    ax.tick_params(**tick_params)  # 'both' refers to minor and major axes

    return ax


def grouped_bar(ax, df: pd.DataFrame, ax_kwargs: dict = {}, legend_kwargs: dict = {}, bar_padding: float = 0.25,
                comp_values=None):
    keys = list(df.index)
    N = len(keys)
    M = len(df.columns)
    x = np.arange(N)  # the label locations

    width = (1 - bar_padding) / M  # the width of the bars

    ax.set_xticks(x + 0.5, minor=True)
    ax.grid(which='minor', axis='x')
    ax.grid(which='major', axis='y')

    # Plotting bars
    for i, col in enumerate(df.columns):
        offset = width * (i) - M / 2 * width + width / 2  # +0*width/2  #- width*N/M #+ 0.125*M - 0.25
        rects = ax.bar(x + offset, df[col], width, label=col, zorder=3)
        # for x_ in x:
        #     ax.axvline(x_,zorder=5,color='k')
        # # ax.bar_label(rects, padding=3)

    ax.set_xticks(x, keys)
    ax.legend(**legend_kwargs)

    ax.set(**ax_kwargs)

    if comp_values is not None:
        for i, x_ in enumerate(x):
            v = comp_values[i]
            ax.plot([x_ - 0.5, x_ + 0.5], [v, v], color='k', zorder=210)

    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    return ax

# %%