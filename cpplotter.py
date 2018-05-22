#!/usr/bin/python
"""This module generates contikipy plots."""
from __future__ import division

import pickle

import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='sans-serif', weight='bold')
plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
plt.rc('text.latex', preamble='\\usepackage{sfmath}')
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=16)

plt.style.use('seaborn-deep')

# Pandas options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ----------------------------------------------------------------------------#
def is_string(obj):
    """Check if an object is a string."""
    return all(isinstance(elem, str) for elem in obj)


# ----------------------------------------------------------------------------#
# Results analysis
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
    # set axis' labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set y limits
    if ylim is not None:
        ax.set_ylim(ylim)
    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # tight layout
    fig.set_tight_layout(False)
    # save  data for post analysis
    if data is not None:
        pickle.dump(data, open(dir + desc + '.pkl', 'w'))

    # save figure
    fig.savefig(dir + 'fig_' + desc + '.pdf')

    # close all open figs
    plt.close('all')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_hist(desc, dir, x, y, **kwargs):
    """Plot a histogram and save."""
    print('> Plotting ' + desc + ' (hist)')
    fig, ax = plt.subplots(figsize=(8, 6))

    # get kwargs
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    ax.hist(x, bins=y, normed=1, histtype='step', cumulative=True,
            stacked=True, fill=True, label=desc)
    # ax.set_xticks(np.arange(0, max(x), 5.0))
    # ax.legend_.remove()

    data = {'x': x, 'y': y,
            'type': 'hist',
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_bar(df, desc, dir, x, y, ylim=None, **kwargs):
    """Plot a barchart and save."""
    print('> Plotting ' + desc + ' (bar)')
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

    data = {'x': x, 'y': y,
            'type': 'bar',
            'width': width,
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_box(desc, dir, x, y, **kwargs):
    """Plot a boxplot and save."""
    print('> Plotting ' + desc + ' (box)')
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

    data = {'x': x, 'y': y,
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
    print('> Plotting ' + desc + ' (violin)')
    fig, ax = plt.subplots(figsize=(8, 6))

    xticks = [0, 1, 2, 3, 4, 5, 6]
    ax.xaxis.set_ticks(xticks)

    ax.violinplot(dataset=[df[df[x] == 1][y],
                           df[df[x] == 2][y],
                           df[df[x] == 3][y],
                           df[df[x] == 4][y]])

    data = {'x': x, 'y': y,
            'type': 'violin',
            'xlabel': xlabel,
            'ylabel': ylabel}

    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_line(df, desc, dir, x, y, **kwargs):
    """Plot a line graph and save."""
    print('> Plotting ' + desc + ' (line)')

    # constants
    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # get kwargs
    errors = kwargs['errors'] if 'errors' in kwargs else None
    steps = kwargs['steps'] if 'steps' in kwargs else 1
    color = kwargs['color'] if 'color' in kwargs else color
    marker = kwargs['marker'] if 'marker' in kwargs else 's'
    ls = kwargs['ls'] if 'ls' in kwargs else '-'
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''
    label = kwargs['label'] if 'label' in kwargs else ylabel

    # set xticks
    xticks = np.arange(min(x), max(x)+1, steps)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if errors is not None:
        ax.errorbar(xticks, y, errors, color=color, marker=marker,
                    ls=ls, lw=2.0)
    else:
        ax.plot(xticks, y, color=color, marker=marker, ls=ls, lw=2.0)

    # legend
    ax.legend(label, loc=2)
    # save figure
    data = {'x': x, 'y': y,
            'type': 'line',
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, dir,
                               xlabel=xlabel, ylabel=ylabel)
    return fig, ax


# ----------------------------------------------------------------------------#
def traffic_ratio(app_df, sdn_df, icmp_df, dir):
    """Plot traffic ratio."""
    # FIXME: Make this generic
    if sdn_df is not None:
        sdn_cbr_len = (sdn_df['typ'] == 'NSU').sum()
        sdn_vbr_len = (sdn_df['typ'] == 'FTQ').sum() + \
                      (sdn_df['typ'] == 'FTS').sum()

    rpl_icmp_count = (icmp_df['type'] == 155).sum()
    if sdn_df is not None:
        total = len(app_df) + sdn_cbr_len + sdn_vbr_len \
                + rpl_icmp_count
    else:
        total = len(app_df) + rpl_icmp_count

    app_ratio = len(app_df)/total  # get app packets as a % of total
    if sdn_df is not None:
        sdn_cbr_ratio = sdn_cbr_len/total  # get sdn packets as a % of total
        sdn_vbr_ratio = sdn_vbr_len/total  # get sdn packets as a % of total
    rpl_icmp_ratio = rpl_icmp_count/total

    if sdn_df is not None:
        df = pd.DataFrame([app_ratio, rpl_icmp_ratio,
                           sdn_cbr_ratio, sdn_vbr_ratio],
                          index=['App', 'RPL', 'SDN-CBR', 'SDN-VBR'],
                          columns=['ratio'])
    else:
        df = pd.DataFrame([app_ratio, rpl_icmp_ratio],
                          index=['App', 'RPL'],
                          columns=['ratio'])
    df.index.name = 'type'
    fig, ax = plot_bar(df, 'traffic_ratio', dir, x=df.index,
                       y=df.ratio)
    data = {'x': df.index, 'y': df.ratio}
    set_fig_and_save(fig, ax, data, 'traffic_ratio', dir,
                     xlabel='Traffic type',
                     ylabel='Ratio of total traffic')

    return fig, ax
