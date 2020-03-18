#!/usr/bin/python
"""This module generates compares results and plots them."""
from __future__ import division

import sys      # for exceptions
import os       # for makedir
import pickle   # for saving data
import re       # for regex
import math
import traceback

import itertools

import matplotlib.pyplot as plt  # general plotting
from matplotlib import mlab
import numpy as np               # number crunching
import scipy.stats as ss
# import seaborn as sns          # fancy plotting
# import pandas as pd            # table manipulation

import cpplotter as cpplot

from ast import literal_eval as make_tuple
from pprint import pprint


marker = itertools.cycle(('s', 'd', 'o', '>', 'x'))
hatch = itertools.cycle(('/', 'x', '\\', '-', '//'))
# ----------------------------------------------------------------------------#
# Helper functions
# ----------------------------------------------------------------------------#
def pad_y(M):
    """Append the minimal amount of zeroes at the end of each array."""
    maxlen = max(len(r) for r in M.values())
    Z = np.zeros((len(M.values()), maxlen))
    i = 0
    for k, v in M.items():
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
    for k, v in M.items():
        if(len(v) > len_x):
            len_x = len(v)
            V = v
    Z = np.zeros((len(M.values()), maxlen))
    i = 0
    for k, v in M.items():
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
def get_unique(dicts, key):
    """Get unique values from a list of dicts given a key."""
    values = []
    for d in dicts:
        if not d[key] in values:
            values.append(d[key])
    return values


# ----------------------------------------------------------------------------#
def merge_data(dicts, key):
    """Merge dicts in a list of dicts based on a key."""
    merged = {}
    for d in dicts:
        d['data']['errors'] = d['data']['errors'].tolist()
        if d[key] in merged:
            merged[d[key]]['data']['x'].extend(d['data']['x'])
            merged[d[key]]['data']['y'].extend(d['data']['y'])
            merged[d[key]]['data']['errors'].extend(d['data']['errors'])
        else:
            merged[d[key]] = d
    return merged


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
                            if (plot + '.pkl') in f and 'ax_' not in f:
                                print('  - found ' + f + ' in ' + sim + '!')
                                d = pickle.load(open(os.path.join(root, sim, f), 'rb'))
                                id = contains_int(sim)
                                plotdata[plot].append({'id': id,
                                                       'label': sim,
                                                       'data': d})
                                found = True
                        if not found:
                            print('- None')
                            raise Exception('ERROR: Can\'t find ' + plot
                                            + '.pkl!')
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

    print(label)
    print(data['y'])


    # mean = np.mean(data['y'])
    # data['errors'] = mean/np.sqrt(data['y'])

    ax.errorbar(data['x'], data['y'], data['errors'],
                color=color, marker=next(marker), ms=12, lw=lw, label=label,
                capsize=3)

    # Re-calculate the xticks
    if any(isinstance(s, str) for s in data['x']):  # check for strings in x
        ind = np.arange(0, len(data['x']), 1)
    else:
        x_min = min(data['x'])
        x_max = max(data['x'])
        ind = np.arange(x_min, x_max + 1, 1)
    ax.set_xticks(ind)


# ----------------------------------------------------------------------------#
def autolabel(ax, rects, max_h):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        rect_height = math.ceil(rect.get_height())
        # rect_height = rect.get_height()
        if rect_height > max_h:
            text_height = rect_height if rect_height <= max_h else max_h
            ax.text(rect.get_x() + rect.get_width()/2., 1*text_height,
                    '%d' % int(rect_height),
                    ha='center', va='bottom')


# ----------------------------------------------------------------------------#
def add_bar(ax, total, index, color, label, data, **kwargs):
    """Add data to bar plot."""
    ylim = kwargs['ylim'] if 'ylim' in kwargs else None
    width = 0.35
    x_len = len(data['x'])
    # check for strings in x
    if not any(isinstance(x, str) for x in data['x']):
        x_max = max(data['x'])
    else:
        x_max = x_len  # if there's a string we use x_len for xticks
    ind = calc_plot_pos(total, index, x_max, x_len)
    rects = ax.bar(ind, data['y'], width, color=color, label=label, edgecolor='black', hatch=next(hatch))
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
    # if there's a y limit then add max value text to top of bars which exceed
    if ylim is not None:
        autolabel(ax, rects, ylim)


# ----------------------------------------------------------------------------#
def add_hist(ax, color, data, bins=30):
    """Add data to histogram plot."""
    ax.semilogx(data['x'], data['y'], 'k-', linewidth=3, color=color, marker=next(marker), ms=12, markevery=0.1)


# ----------------------------------------------------------------------------#
def compare_box(ax, datasets, **kwargs):
    """Compare box plots."""
    legend = kwargs['legend'] if 'legend' in kwargs else 'best'
    labels = []             # save the labels for the legend
    history = []            # save the data history
    artists = []            # save the artists for the legend
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
    if legend:
        labels = [label.replace("uSDN", "$\\mu$SDN") for label in labels]
        ax.legend(artists, labels, loc=legend)

    # boxplot_zoom(ax, lastdata,
    #              width=1.5, height=1.5,
    #              xlim=[0, 6.5], ylim=[0, 11000],
    #              bp_width=width, pos=[5])
    return ax


# ----------------------------------------------------------------------------#
def compare_bar(ax, datasets, **kwargs):
    """Compare bar plots."""
    legend = kwargs['legend'] if 'legend' in kwargs else 'best'
    ylim = kwargs['ylim'] if 'ylim' in kwargs else None
    # variables for later
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
    # print(Y)
    Y = pad_y(Y)
    # set a y limit
    # print(ylim)
    if ylim is not None:
        ax.set_ylim([0, ylim])
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
                data['label'], data['data'], ylim=ylim)
        # increment plot index
        index += 1

    # add legend
    if legend:
        labels = [label.replace("uSDN", "$\\mu$SDN") for label in labels]
        ax.legend(labels, loc=legend)

    return ax


# ----------------------------------------------------------------------------#
def compare_line(ax, datasets, **kwargs):
    """Compare line plots."""
    legend = kwargs['legend'] if 'legend' in kwargs else 'best'
    labels = []             # save the labels for the legend
    # history = []            # save the data history
    index = 1   # start index
    # amalgamate the x ticks
    ids = get_unique(datasets, 'id')
    labels = get_unique(datasets, 'label')
    datasets = merge_data(datasets, 'id')
    for data in datasets:
        index = ids.index(data)
        data = datasets[data]
        print('      > Compare line fig ... ' + str(index))
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index]['color']  # NB plt.rcParams['axes.prop_cycle'] only goes up to 6!
        # plot the line and add to the parent fig
        add_line(ax, color, data['label'], data['data'])
    # add legend
    if legend:
        labels = [label.replace("uSDN", "$\\mu$SDN") for label in labels]
        ax.legend(labels, loc=legend)

    ax.set_yscale('log')
    ax.set_ylim(pow(10, 0.5), pow(10, 3.5))

    return ax


# ----------------------------------------------------------------------------#
def compare_hist(ax, datasets, **kwargs):
    """Compare histograms."""
    legend = kwargs['legend'] if 'legend' in kwargs else 'best'
    labels = []             # save the labels for the legend
    history = []            # save the data history
    index = 1   # start index
    # compare each dataset for this plot
    for data in datasets:
        history.append(data['data'])
        labels.append(data['label'])
        # Set the color for this iteration color (cyclic)
        color = list(plt.rcParams['axes.prop_cycle'])[index-1]['color']
        # plot the line and add to the parent fig
        add_hist(ax, color, data['data'])
        # increment plot index
        index += 1

    # add legend
    if legend:
        labels = [label.replace("uSDN", "$\\mu$SDN") for label in labels]
        ax.legend(labels, loc=legend)

    ax.set_xlim(pow(10, -1.5), pow(10, 2.5))

    return ax


# ----------------------------------------------------------------------------#
def compare(dir, simlist, plottypes, args, **kwargs):
    """Compare results between data sets for a list of plot types."""
    # default values
    legend = 'best'
    ylim = None
    # dictionary of the various plot comparison functions
    function_map = {
        'bar':   compare_bar,
        'box':   compare_box,
        'line':  compare_line,
        'hist':  compare_hist,
    }
    # Set defaults
    samefigure = 0
    nrows = 1
    ncols = 1
    # search for the required plots
    plotdata = search_dirs(dir, simlist, plottypes)
    if args is not None:
        if 'nrows' in args and 'ncols' in args:
            nrows = args['nrows']
            ncols = args['ncols']
        if 'samefigure' in args:
            samefigure = args['samefigure']
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12),
                                     sharex=True)
    # loop through the sims, comparing the data in the datasets and then plot
    for plot, datasets in plotdata.items():
        print('> Compare ' + str(len(datasets)) + ' datasets for ' + plot, end=' ')
        # check all the types, xlabels and ylabels match for entries in the ds
        # TODO: Throw if not
        for data in datasets:
            datatype = data['data']['type']
            xlabel = data['data']['xlabel']
            ylabel = data['data']['ylabel']
        print('(' + str(datatype).upper() + ') ...'),

        # check for sim arguments
        if args is not None and plot in args:
            print(plot)
            # check for specified legend position
            legend = args[plot]['legend'] if 'legend' in args[plot] else legend
            bbox_pattern = re.compile('\\(.*\\)')
            if legend == 'None':
                legend = None
            elif bbox_pattern.match(legend):
                legend = make_tuple(legend)
            # check for row and col
            row = args[plot]['row']
            col = args[plot]['col']
            # check for a y limit
            ylim = args[plot]['ylim'] if 'ylim' in args[plot] else None

        if samefigure == 1:
            print('SAME FIG'),
            if nrows != 1 or ncols != 1:
                if nrows > 1 and ncols == 1:
                    ax = axes[row]
                elif nrows == 1 and ncols > 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
            else:
                ax = axes
            # call the function map
            ax = function_map[datatype](ax, datasets, legend=legend, ylim=ylim)
            # HACK: Set the correct labels if same figure
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Remove top axes and right axes ticks
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        # DIFFERENT FIGURESargs[sim] else legend
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            ax = function_map[datatype](axes, datasets, legend=legend, ylim=ylim)
            cpplot.set_fig_and_save(fig, ax, None,
                                    plot + '_' + str(simlist),  # filename
                                    dir + '/',                 # directory
                                    xlabel=xlabel,
                                    ylabel=ylabel)

    # save if all on same figure
    if samefigure == 1:
        cpplot.set_fig_and_save(fig, None, None,
                                plot + '_' + str(plottypes),  # filename
                                dir + '/')                   # directory

    print('> SUCCESS! Finshed comparing plots :D')
