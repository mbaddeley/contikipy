#!/usr/bin/python
"""This module generates compares results and plots them."""
from __future__ import division

import sys      # for exceptions
import os       # for makedir
import pickle   # for saving data
import re       # for regex
import traceback

import matplotlib.pyplot as plt  # general plotting
import numpy as np               # number crunching
# import seaborn as sns          # fancy plotting
# import pandas as pd            # table manipulation

from pprint import pprint

import cpplotter as cpplot


# ----------------------------------------------------------------------------#
# Helper functions
# ----------------------------------------------------------------------------#
def pad_y(M):
    """Append the minimal amount of zeroes at the end of each array."""
    maxlen = max(len(r) for r in M.values())
    Z = np.zeros((len(M.values()), maxlen))
    i = 0
    for k, v in M.iteritems():
        Z[i, :len(v)] += v
        M[k] = Z[i]
        i = i + 1
    return M


# ----------------------------------------------------------------------------#
def pad_x(M):
    """Pad each array with incremental values."""
    maxlen = max(len(r) for r in M.values())
    len_x = 0
    V = []
    for k, v in M.iteritems():
        if(len(v) > len_x):
            len_x = len(v)
            V = v
    Z = np.zeros((len(M.values()), maxlen))
    i = 0
    for k, v in M.iteritems():
        Z[i, :len(v)] += v
        Z[i, len(v):maxlen] = V[len(v):maxlen]
        M[k] = Z[i]
        i = i + 1
    return M


# ----------------------------------------------------------------------------#
def contains_int(string):
    """Return the first integer number found in a string."""
    match = re.search('\\d+', string)
    if match is None:
        return string
    else:
        return int(match.group())


# ----------------------------------------------------------------------------#
# Main functions
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
                for sim in simlist:
                    if sim in sorted(dirs):
                        found = False
                        print('  ... Scanning \"' + root + '/' + sim + '/\"')
                        for f in os.listdir(os.path.join(root, sim)):
                            if (plot + '.pkl') == f:
                                print('  - found ' + plot + '.pkl in '
                                      + sim + '!')
                                d = pickle.load(open(os.path.join(root,
                                                                  sim, f)))
                                id = contains_int(sim)
                                plotdata[plot].append({'id': id,
                                                       'label': sim,
                                                       'data': d})
                                found = True
                        if not found:
                            print('- None')
                            raise Exception('ERROR: Can\'t find ' + plot +
                                            '.pkl!')
    except Exception as e:
            traceback.print_exc()
            print(e)
            sys.exit(0)
    # pprint(plotdata)
    return plotdata


def calc_plot_pos(total_plots, plot_index, x_max, x_len, gap=0.35, width=0.35):
    """Calculate an array of x positions based on plot index."""
    ind = []
    i = plot_index-1
    start = gap
    for n in range(x_len):
        pos = start + n*gap + i*width + n*(total_plots*width)
        ind.append(pos)
    return ind


def calc_xtick_pos(total_plots, plot_index, x_max, x_len,
                   gap=0.35, width=0.35):
    """Calculate the midpoint positions for xticks, based on plot index."""
    ind = []
    start = gap
    for n in range(x_len):
        pos = start/2 + n*gap + n*(total_plots*width) + (total_plots*width)/2
        ind.append(pos)
    return ind


# ----------------------------------------------------------------------------#
def add_box(ax, artists, total, index, color, label, data):
    """Add data to box plot."""
    width = 0.35
    notch = False
    fliers = False
    x_max = max(data['x'])
    x_len = len(data['x'])
    ind = calc_plot_pos(total, index, x_max, x_len)
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
    ind = calc_xtick_pos(total, index, x_max, x_len)
    ax.set_xticks(ind)
    ax.set_xticklabels(data['x'])


# ----------------------------------------------------------------------------#
def add_line(ax, color, label, data, **kwargs):
    """Add data to bar plot."""
    lw = 4.0
    marker = 's'
    x_min = min(data['x'])
    x_max = max(data['x'])

    ax.errorbar(data['x'], data['y'], data['errors'],
                color=color, marker=marker, lw=lw, label=label,
                capsize=3)

    # Re-calculate the xticks
    ind = np.arange(x_min, x_max + 1, 1)
    ax.set_xticks(ind)


# ----------------------------------------------------------------------------#
def add_bar(ax, total, index, color, label, data):
    """Add data to bar plot."""
    width = 0.35
    x_len = len(data['x'])
    # check for strings in x
    if not any(isinstance(x, str) for x in data['x']):
        x_max = max(data['x'])
    else:
        x_max = x_len  # if there's a string we use x_len for xticks
    ind = calc_plot_pos(total, index, x_max, x_len)
    ax.bar(ind, data['y'], width, color=color, label=label)
    # Re-calculate the xticks
    ind = calc_xtick_pos(total, index, x_max, x_len)
    ax.set_xticks(ind)
    # convert xlabels to ints if they are floats
    xlabels = [str(int(x)) if isinstance(x, float) else x for x in data['x']]
    # xlabels = [r'\textbf{' + x + '}' for x in xlabels]  # bold
    # make font smaller if data['x'] contains strings
    if any(isinstance(x, str) for x in data['x']):
        rc_params = {'fontsize': 12}
    else:
        rc_params = None
    ax.set_xticklabels(xlabels, rc_params)


# ----------------------------------------------------------------------------#
def add_hist(ax, color, data, bins=30):
    """Add data to histogram plot."""
    # FIXME: Currently an issue with outliers causing smaller plots to be
    #       unreadable. Using range() in mean time.
    norm = 1
    range = (0, 50)
    type = 'bar'
    cumul = True
    stack = True
    fill = True
    x = sorted(data)
    bins = np.around(np.linspace(0, max(x), len(x)), 3)  # bin values to 3dp
    if bins is None:
        bins = x
    (n, bins, patches) = ax.hist(x, bins=bins,
                                 normed=norm,
                                 histtype=type,
                                 cumulative=cumul,
                                 stacked=stack,
                                 fill=fill,
                                 color=color)
    # pprint(bins)


# ----------------------------------------------------------------------------#
def compare_box(ax, datasets, **kwargs):
    """Compare box plots."""
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
        # plot the box and add to the parent fig
        add_box(ax, artists, len(datasets), index, color,
                data['label'], data['data'])
        # increment plot index
        index += 1

    # add legend (need to do this here because of the artists)
    ax.legend(artists, labels, loc='best')

    # boxplot_zoom(ax, lastdata,
    #              width=1.5, height=1.5,
    #              xlim=[0, 6.5], ylim=[0, 11000],
    #              bp_width=width, pos=[5])
    return ax


# ----------------------------------------------------------------------------#
def compare_bar(ax, datasets, **kwargs):
    """Compare bar plots."""
    labels = []             # save the labels for the legend
    history = []            # save the data history
    index = 1   # start index
    # compare each dataset for this plot
    X = {}
    Y = {}
    # pad the datasets with zeros up to the size of the largest x/y
    for data in datasets:
        X[data['id']] = data['data']['x']
        Y[data['id']] = data['data']['y']
    if not any(isinstance(x, str) for x in data['data']['x']):
        X = pad_x(X)
    Y = pad_y(Y)
    # plot the data
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # plot the bar and add to the parent fig
        data['data']['x'] = X[data['id']]
        data['data']['y'] = Y[data['id']]
        add_bar(ax, len(datasets), index, color,
                data['label'], data['data'])
        # increment plot index
        index += 1

    # add legend
    # ax.legend(labels, loc='best')
    ax.legend(labels, loc='best', bbox_to_anchor=(1, 1))
    # ax.legend(labels, loc='lower center', ncol=n_plots)
    return ax


# ----------------------------------------------------------------------------#
def compare_line(ax, datasets, **kwargs):
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
        # plot the line and add to the parent fig
        add_line(ax, color, data['label'], data['data'])
        # increment plot index
        index += 1

    # add legend
    ax.legend(labels, loc='best')
    # ax.legend(labels, loc='best', bbox_to_anchor=(1, 1))
    return ax


# ----------------------------------------------------------------------------#
def compare_hist(ax, datasets, **kwargs):
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
        # plot the line and add to the parent fig
        add_hist(ax, color, data['data']['x'])
        # increment plot index
        index += 1

    # add legend
    ax.legend(labels, loc='best')
    # ax.legend(['RPL-DAG', r'$\mu$SDN-Controller'],
    #           loc='lower right')
    return ax


# ----------------------------------------------------------------------------#
def compare(dir, simlist, plottypes, args, **kwargs):
    """Compare results between data sets for a list of plot types."""
    # dictionary of the various comparison functions
    function_map = {
        'bar':   compare_bar,
        'box':   compare_box,
        'line':  compare_line,
        'hist':  compare_hist,
    }

    # search for the required plots
    plotdata = search_dirs(dir, simlist, plottypes)
    for plot, datasets in plotdata.items():
        print('> Compare ' + str(len(datasets)) + ' datasets for ' + plot),
        # sort the datasets for each plot
        # datasets = sorted(datasets, key=lambda d: d['id'], reverse=False)
        # check all the dataset types, xlabels and ylabels match
        # TODO: Throw if not
        for data in datasets:
            type = data['data']['type']
            xlabel = data['data']['xlabel']
            ylabel = data['data']['ylabel']
        print('(' + str(type).upper() + ') ...'),

        # call appropriate comparison function and plot
        # if 'samefigure' in args and args['samefigure'] is 1:
        fig, axes = plt.subplots(args['nrows'], args['ncols'], figsize=(8, 6))
        print(isinstance(axes, list))
        pprint(axes)
        if (isinstance(axes, list)):
            row = args[plot]['row']
            col = args[plot]['col']
            # x = np.linspace(0, 2*np.pi, 400)
            # y = np.sin(x**2)
            ax = function_map[type](axes[0, 0], datasets)
        else:
            ax = function_map[type](axes, datasets)

        # make labels bold
        # labels = [r'\textbf{' + label + '}' for label in labels]
        # add escape for underscores
        # labels = [label.replace("_", "\\_") for label in labels]
        # ax.legend(labels, loc='best')
        # ax.legend(labels, loc='lower center', ncol=n_plots)

        # save figure
        cpplot.set_fig_and_save(fig, ax, None,
                                plot + '_' + str(simlist),  # filename
                                dir + '/',                  # directory
                                xlabel=xlabel,
                                ylabel=ylabel)
        print('OK')

    print('> SUCCESS! Finshed comparing plots :D')
