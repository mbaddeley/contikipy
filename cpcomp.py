#!/usr/bin/python
"""This module generates compares results and plots them."""
from __future__ import division

import os  # for makedir
import pickle
import re  # regex

import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation

from pprint import pprint

import cpplotter as cpplot

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='sans-serif', weight='bold')
plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
plt.rc('text.latex', preamble='\usepackage{sfmath}')
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
def contains_int(string):
    """Return the first integer number found in a string."""
    match = re.search('\d+', string)
    if match is None:
        return string
    else:
        return int(match.group())


# ----------------------------------------------------------------------------#
def search_dirs(rootdir, simlist, plottypes):
    """Search simulation folders to collate data."""
    plotdata = {}  # dictionary of plot data
    # for each plot type
    for plot in plottypes:
        plotdata[plot] = []  # create a list to hold data structs
        print '> Looking for plots of type ... ' + plot
        # walk through directory structure
        for root, dirs, files in os.walk(rootdir):
            for dir in sorted(dirs):
                print dir
                print simlist
                if dir in simlist:
                    found = False
                    print ' ... Scanning \"' + root + '/' + dir + '/\"',
                    for f in os.listdir(os.path.join(root, dir)):
                        if (plot + '.pkl') in f:
                            print '- found pickle in ' + dir + '!'
                            d = pickle.load(file(os.path.join(root, dir, f)))
                            id = contains_int(dir)
                            plotdata[plot].append({'id': id,
                                                   'label': dir,
                                                   'data': d})
                            found = True
                    if not found:
                        print '- None'

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
    ax.plot(data['x'], data['y'],
            color=color, marker=marker, lw=lw, label=label)


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
def add_hist(ax, color, data):
    """Add data to histogram plot."""
    norm = 1
    type = 'step'
    cumul = True
    stack = True
    fill = True
    ax.hist(data['x'], bins=data['y'],
            normed=norm, histtype=type,
            cumulative=cumul, stacked=stack, fill=fill,
            color=color)


# ----------------------------------------------------------------------------#
def compare(dir, simlist, plottypes, **kwargs):
    """Compare results between data sets for a list of plot types."""
    print '*** Analyzing (comparing) results in dir: ' + dir
    print '* Comparing simulations: [' + ', '.join(simlist) + ']'
    print '* Generating plots: [' + ', '.join(plottypes) + ']'

    gap = 0.5  # gap for xticks
    width = 0.35

    xmax = None  # work out xmax

    plotdata = search_dirs(dir, simlist, plottypes)
    # iterate over all the plots we have data for

    # pprint(plotdata.items())
    # for each plot type where we have found sims to compare
    for plottype, sims in plotdata.items():
        count = 1               # reset sim counter
        nsims = len(sims)       # number sims to compare
        artists = []            # save the artists for the legend
        labels = []             # save the labels for the legend
        # create a new figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # sort the data
        sims = sorted(sims, key=lambda d: d['id'], reverse=False)

        print '> Compare ' + str(nsims) + ' plots of type \'' + plottype + '\''
        # for each sim which has this plot type
        for sim in sims:
            data = sim['data']
            label = sim['label']

            # save the label for the legend
            labels.append(label)
            # Set the color for this iteration color (cyclic)
            color = list(plt.rcParams['axes.prop_cycle'])[count-1]['color']

            # print some info about this simulation
            print ' ... ' + label + ' ' + data['type'] + ' plot',
            print '(' + str(count) + '/' + str(nsims) + ') color=' + color

            # Add the data to the figure
            if data['type'] == 'box':
                add_box(ax, artists, count, label, data)
            elif data['type'] == 'line':
                add_line(ax, color, label, data)
            elif data['type'] == 'bar':
                add_bar(ax, count, color, label, data)
            elif data['type'] == 'hist':
                add_hist(ax, color, data)
            else:
                print 'Error: no type \'' + data['type'] + '\''

            # increment plot count
            count += 1

        # We have gone over each simulation type. Set a few more params
        if data['type'] == 'line':
            xticks = np.arange(min(data['x']), max(data['x'])+1, 1)
            ax.set_xticks(xticks)

        # legend
        for label in labels:
            label = r'\textbf{' + label + '}'  # make labels bold
        if artists:
            ax.legend(artists, labels, loc='best')
            # ax.set_xticks([1, 2, 3])
            # ax.set_xticklabels(['180', '300', '600'])
        else:
            if 'hops_prr' in plottype:
                ax.legend(labels, loc='best')
            elif 'hops_rdc' in plottype:
                ax.legend(labels, loc='lower right')
            elif 'join' in plottype:
                ax.legend(['RPL-DAG', r'$\mu$SDN-Controller'],
                          loc='lower right')
            else:
                ax.legend(labels, loc='best')
        # boxplot_zoom(ax, lastdata,
        #              width=1.5, height=1.5,
        #              xlim=[0, 6.5], ylim=[0, 11000],
        #              bp_width=width, pos=[5])

        # save figure
        cpplot.set_fig_and_save(fig, ax, None,
                                plottype + '_' + str(simlist),  # filename
                                dir + '/',  # directory
                                xlabel=data['xlabel'],
                                ylabel=data['ylabel'])
    print '*** Finished analysis!'
