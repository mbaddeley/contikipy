#!/usr/bin/python
"""This module generates compares results and plots them."""
from __future__ import division

import sys      # for exceptions
import os       # for makedir
import pickle   # for saving data
import re       # for regex

import matplotlib.pyplot as plt  # general plotting
import numpy as np               # number crunching
# import seaborn as sns          # fancy plotting
import pandas as pd              # table manipulation

from pprint import pprint

import cpplotter as cpplot

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='sans-serif', weight='bold')
rc('xtick', labelsize=18)
rc('ytick', labelsize=18)
rc('axes', labelsize=18)
rc('legend', fontsize=16)
# plt.rc('text.latex', preamble='\\usepackage{sfmath}')

plt.style.use('seaborn-deep')

# Pandas options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ----------------------------------------------------------------------------#
def contains_int(string):
    """Return the first integer number found in a string."""
    match = re.search('\\d+', string)
    if match is None:
        return string
    else:
        return int(match.group())


# ----------------------------------------------------------------------------#
def search_dirs(rootdir, simlist, plottypes):
    """Search simulation folders to collate data."""
    plotdata = {}  # dictionary of plot data
    # for each plot type
    print('> Looking for plots in ' + str(simlist))
    try:
        for plot in plottypes:
            plotdata[plot] = []  # create a list to hold data structs
            print('> Plot type: ' + plot + ' ...')
            # walk through directory structure
            for root, dirs, files in os.walk(rootdir):
                for dir in sorted(dirs):
                    if dir in simlist:
                        found = False
                        print('  ... Scanning \"' + root + '/' + dir + '/\"')
                        for f in os.listdir(os.path.join(root, dir)):
                            if (plot + '.pkl') in f:
                                print('  - found ' + plot + '.pkl in '
                                      + dir + '!')
                                d = pickle.load(open(os.path.join(root,
                                                                  dir, f)))
                                id = contains_int(dir)
                                plotdata[plot].append({'id': id,
                                                       'label': dir,
                                                       'data': d})
                                found = True
                        if not found:
                            print('- None')
                            raise Exception('ERROR: Can\'t find ' + plot +
                                            '.pkl!')
    except Exception as e:
            print(e)
            sys.exit(0)

    return plotdata


def calc_plot_pos(plot_index, x_max, x_len, gap=0, width=0.35):
    """Calculate an array of x positions based on plot index."""
    n = plot_index-1
    start = width*(n) + gap*(n)
    end = x_max + start
    step = (end - start) / x_len
    return np.arange(start, end, step)


def calc_xtick_pos(plot_index, x_max, x_len, gap=0, width=0.35):
    """Calculate the midpoint positions for xticks, based on plot index."""
    n = plot_index-1
    start = width*(n)/plot_index + gap*(n)/plot_index
    end = x_max + start
    step = (end - start) / x_len
    return np.arange(start, end, step)


# ----------------------------------------------------------------------------#
def add_box(ax, artists, index, label, data):
    """Add data to box plot."""
    width = 0.35
    notch = False
    fliers = False
    x_max = max(data['x'])
    x_len = len(data['x'])
    ind = calc_plot_pos(index, x_max, x_len, gap=0.1)
    bp = ax.boxplot(data['y'],
                    positions=ind,
                    notch=notch,
                    widths=width,
                    showfliers=fliers,
                    patch_artist=True,
                    manage_xticks=False)
    cpplot.set_box_colors(bp, index-1)
    artists.append(bp["boxes"][0])
    # Re-calculate the xticks
    ind = calc_xtick_pos(index, x_max, x_len, gap=0.1)
    ax.set_xticks(ind)
    ax.set_xticklabels(data['x'])


# ----------------------------------------------------------------------------#
def add_line(ax, color, label, data):
    """Add data to bar plot."""
    lw = 2.0
    marker = 's'
    x_min = min(data['x'])
    x_max = max(data['x'])
    ax.plot(data['x'], data['y'],
            color=color, marker=marker, lw=lw, label=label)
    # Re-calculate the xticks
    ind = np.arange(x_min, x_max + 1, 1)
    ax.set_xticks(ind)


# ----------------------------------------------------------------------------#
def add_bar(ax, index, color, label, data):
    """Add data to bar plot."""
    width = 0.35
    x_max = max(data['x'])
    x_len = len(data['x'])
    ind = calc_plot_pos(index, x_max, x_len)
    ax.bar(ind, data['y'], width, color=color, label=label)
    # Re-calculate the xticks
    ind = calc_xtick_pos(index, x_max, x_len)
    ax.set_xticks(ind)
    ax.set_xticklabels(data['x'])


# ----------------------------------------------------------------------------#
def add_hist(ax, color, data, bins=30):
    """Add data to histogram plot."""
    # FIXME: Currently an issue with outliers causing smaller plots to be
    #       unreadable. Using range() in mean time.
    norm = 1
    bins = 30
    range = (0, 50)
    type = 'bar'
    cumul = True
    stack = False
    fill = True
    x = sorted(data)
    if bins is None:
        bins = x
    (n, bins, patches) = ax.hist(x, bins=bins, range=range,
                                 normed=norm,
                                 histtype=type,
                                 cumulative=cumul,
                                 stacked=stack,
                                 fill=fill,
                                 color=color)
    # pprint(bins)


# ----------------------------------------------------------------------------#
def compare_box(datasets, **kwargs):
    """Compare box plots."""
    print('> Compare box plots')
    labels = []             # save the labels for the legend
    history = []            # save the data history
    artists = []            # save the artists for the legend
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))    # create a new figure
    index = 1   # start index
    # compare each dataset for this plot
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # print(some info about this simulation
        # pprint(data)
        # plot the box and add to the parent fig
        add_box(ax, artists, index, color, data['label'], data['data'])
        # increment plot index
        index += 1

    ax.legend(artists, labels, loc='best')

    # boxplot_zoom(ax, lastdata,
    #              width=1.5, height=1.5,
    #              xlim=[0, 6.5], ylim=[0, 11000],
    #              bp_width=width, pos=[5])

    return fig, ax, labels


# ----------------------------------------------------------------------------#
def compare_bar(datasets, **kwargs):
    """Compare bar plots."""
    labels = []             # save the labels for the legend
    history = []            # save the data history
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))    # create a new figure
    index = 1   # start index
    # compare each dataset for this plot
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # print(some info about this simulation
        # pprint(data)
        # plot the bar and add to the parent fig
        add_bar(ax, index, color, data['label'], data['data'])
        # increment plot index
        index += 1

    ax.legend(labels, loc='best')

    return fig, ax, labels


# ----------------------------------------------------------------------------#
def compare_line(datasets, **kwargs):
    """Compare line plots."""
    labels = []             # save the labels for the legend
    history = []            # save the data history
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))    # create a new figure
    index = 1   # start index
    # compare each dataset for this plot
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # print(some info about this simulation
        # pprint(data)
        # plot the line and add to the parent fig
        add_line(ax, color, data['label'], data['data'])
        # increment plot index
        index += 1

    ax.legend(labels, loc='best')


# ----------------------------------------------------------------------------#
def compare_hist(datasets, **kwargs):
    """Compare histograms."""
    labels = []             # save the labels for the legend
    history = []            # save the data history
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))    # create a new figure
    index = 1   # start index
    # compare each dataset for this plot
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # print(some info about this simulation
        # pprint(data)
        # plot the line and add to the parent fig
        add_hist(ax, color, data['x'])
        # increment plot index
        index += 1

    # ax.legend(['RPL-DAG', r'$\mu$SDN-Controller'],
    #           loc='lower right')
    ax.legend(labels, 'lower right')


# ----------------------------------------------------------------------------#
def compare(dir, simlist, plottypes, **kwargs):
    """Compare results between data sets for a list of plot types."""
    print('*** Compare plots in dir: ' + dir)
    # print(' ... simulations: ' + str(simlist))
    # print(' ... plots: ' + str(plottypes))

    # dictionary of the various comparison functions
    function_map = {
        'bar':   compare_bar,
        'box':   compare_box,
        'line':  compare_line,
        'hist':  compare_hist,
    }

    plotdata = search_dirs(dir, simlist, plottypes)
    for plot, datasets in plotdata.items():
        print('> Compare ' + str(len(datasets)) + ' datasets for ' + plot),
        # sort the datasets for each plot
        datasets = sorted(datasets, key=lambda d: d['id'], reverse=False)

        try:
            # check all the dataset types, xlabels and ylabels match
            for data in datasets:
                type = data['data']['type']
                xlabel = data['data']['xlabel']
                ylabel = data['data']['ylabel']
            print('... (' + type.upper() + ')')
            # call appropriate comparison function
            fig, ax, labels = function_map[type](datasets)
            # make labels bold
            for label in labels:
                label = r'\textbf{' + label + '}'
            # save figure
            cpplot.set_fig_and_save(fig, ax, None,
                                    plot + '_' + str(simlist),  # filename
                                    dir + '/',                  # directory
                                    xlabel=xlabel,
                                    ylabel=ylabel)
            print('  ... OK')
        except Exception as e:
                print(e)
                sys.exit(0)
    print('> SUCCESS! Finshed comparing plots :D')
