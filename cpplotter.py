#!/usr/bin/python
"""This module generates contikipy plots."""
from __future__ import division

import os
import pickle

import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
# import pandas as pd  # table manipulation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='sans-serif', weight='bold')
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble='\\usepackage{sfmath}')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=20, labelweight='bold')
plt.rc('legend', fontsize=14)

plt.style.use('seaborn-deep')


# ----------------------------------------------------------------------------#
def is_string(obj):
    """Check if an object is a string."""
    return all(isinstance(elem, str) for elem in obj)


# ----------------------------------------------------------------------------#
# Results compare
# ----------------------------------------------------------------------------#
def set_box_colors(bp, index):
    """Set the boxplot colors."""
    color = list(plt.rcParams['axes.prop_cycle'])[index]['color']
    # lw = 1.5
    for box in bp['boxes']:
        # change fill color
        box.set(facecolor=color)
    # change color the medians
    for median in bp['medians']:
        median.set(color='black')
    # for box in bp['boxes']:
    #     # change outline color
    #     box.set(color='#7570b3', linewidth=linewidth)
    #     # # change fill color
    #     box.set(facecolor='b')
    # # change color and linewidth of the whiskers
    # for whisker in bp['whiskers']:
    #     whisker.set(color='#7570b3', linewidth=linewidth)
    # # change color and linewidth of the caps
    # for cap in bp['caps']:
    #     cap.set(color='#7570b3', linewidth=linewidth)
    # # change the style of fliers and their fill
    # for flier in bp['fliers']:
    #     flier.set(marker='o', markerfacecolor='#e7298a', alpha=0.5)


# ----------------------------------------------------------------------------#
def boxplot_zoom(ax, data, width=1, height=1,
                 xlim=None, ylim=None,
                 bp_width=0.35, pos=[1], color=1):
    """Plot a zoomed boxplot and save."""
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    axins = inset_axes(ax, width, height, loc=1)
    bp = axins.boxplot(data, notch=False, positions=pos, widths=bp_width,
                       showfliers=False, patch_artist=True)
    set_box_colors(bp, color)
    axins.axis([4.7, 5.3, 310, 370])
    axins.set_xticks([])
    mark_inset(ax, axins,
               loc1=3, loc2=4,  # left and right line anchors
               fc="none",  # facecolor
               ec="0.3",  # edgecolor
               ls='--')


# ----------------------------------------------------------------------------#
def set_fig_and_save(fig, ax, data, desc, dir, **kwargs):
    """Set figure properties and save as pdf."""
    # get kwargs
    ylim = kwargs['ylim'] if 'ylim' in kwargs else None
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    if ax is not None:
        # set y limits
        ax.set_ylim(ylim)
        # set axis' labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # tight layout
    fig.set_tight_layout(False)

    # save  data for post compare
    os.makedirs(dir, exist_ok=True)
    if data is not None:
        pickle.dump(data, open(dir + desc + '.pkl', 'wb'))
    # save ax for post compare
    pickle.dump(ax, open(dir + 'ax_' + desc + '.pkl', 'wb+'))
    # save pdf of figure plus the figure itself
    fig.savefig(dir + 'fig_' + desc + '.pdf', bbox_inches="tight")

    # with open('myplot.pkl','rb') as fid:
    # ax = pickle.load(fid)

    # close all open figs
    plt.close('all')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_hist(df, desc, dir, x, y, ylim=None, **kwargs):
    """Plot a histogram and save."""
    print('> Plotting ' + desc + ' (HIST)')
    fig, ax = plt.subplots(figsize=(8, 6))

    # get kwargs
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    bins = np.around(np.linspace(0, max(x), len(x)), 3)  # bin values to 3dp
    ax.hist(x, bins, density=1, histtype='step', cumulative=True,
            stacked=True, fill=True, label=desc, color=color)
    # ax.set_xticks([bins[0], bins[len(x)-1]])
    ax.set_xticks(np.linspace(bins[0], bins[len(bins)-1], 5))
    # ax.legend_.remove()

    data = {'x': x, 'y': y, 'errors': None,
            'type': 'hist',
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_bar(df, desc, dir, x, y, ylim=None, **kwargs):
    """Plot a barchart and save."""
    print('> Plotting ' + desc + ' (BAR)')
    fig, ax = plt.subplots(figsize=(8, 6))

    # constants
    width = 0.35  # the width of the bars
    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # get kwargs
    color = kwargs['color'] if 'color' in kwargs else color
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    ind = np.arange(len(x))

    ax.bar(x=ind, height=y, width=width, color=color)

    # set x-axis
    ax.set_xticks(np.arange(min(ind), max(ind)+1, 1.0))
    # check for string, if not then convert x to ints for the label
    if not is_string(x):
        x = [int(i) for i in x]
    ax.set_xticklabels(x)
    # set y limits
    if ylim is not None:
        ax.set_ylim(ylim)

    data = {'x': x, 'y': y, 'errors': None,
            'type': 'bar',
            'width': width,
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_box(df, desc, dir, x, y, ylim=None, **kwargs):
    """Plot a boxplot and save."""
    print('> Plotting ' + desc + ' (BOX)')
    # subfigures
    fig, ax = plt.subplots(figsize=(8, 6))

    # constants
    # ylim = [0, 1500]
    width = 0.5   # the width of the boxes
    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # get kwargs
    color = kwargs['color'] if 'color' in kwargs else color
    ylim = kwargs['ylim'] if 'ylim' in kwargs else None
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    # Filter data using np.isnan
    mask = ~np.isnan(y)
    y = [d[m] for d, m in zip(y.T, mask.T)]
    bp = ax.boxplot(y, showfliers=False, patch_artist=True)
    set_box_colors(bp, 0)

    data = {'x': x, 'y': y, 'errors': None,
            'type': 'box',
            'width': width,
            'xlabel': xlabel,
            'ylabel': ylabel}

    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               ylim=ylim,
                               xlabel=xlabel,
                               ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_violin(df, desc, dir, x, xlabel, y, ylabel):
    """Plot a violin plot and save."""
    print('> Plotting ' + desc + ' (VIOLIN)')
    fig, ax = plt.subplots(figsize=(8, 6))

    xticks = [0, 1, 2, 3, 4, 5, 6]
    ax.xaxis.set_ticks(xticks)

    ax.violinplot(dataset=[df[df[x] == 1][y],
                           df[df[x] == 2][y],
                           df[df[x] == 3][y],
                           df[df[x] == 4][y]])

    data = {'x': x, 'y': y, 'errors': None,
            'type': 'violin',
            'xlabel': xlabel,
            'ylabel': ylabel}

    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_line(df, desc, dir, x, y, **kwargs):
    """Plot a line graph and save."""
    print('> Plotting ' + desc + ' (LINE)')

    # constants
    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # get kwargs
    errors = kwargs['errors'] if 'errors' in kwargs else None
    steps = kwargs['steps'] if 'steps' in kwargs else 1
    color = kwargs['color'] if 'color' in kwargs else color
    marker = kwargs['marker'] if 'marker' in kwargs else 's'
    ls = kwargs['ls'] if 'ls' in kwargs else '-'
    lw = kwargs['lw'] if 'lw' in kwargs else 2.0
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
    # label = kwargs['label'] if 'label' in kwargs else ylabel

    # set xticks
    xticks = np.arange(min(x), max(x)+1, steps)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xticks, y, errors, color=color, marker=marker, ls=ls, lw=lw)

    # ax.legend(label, loc=2)
    # save figure
    data = {'x': x, 'y': y, 'errors': errors,
            'type': 'line',
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)
    return fig, ax
