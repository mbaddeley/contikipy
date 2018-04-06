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

# from pprint import pprint

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
    print match
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


# ----------------------------------------------------------------------------#
def compare_results(rootdir, simlist, plottypes, **kwargs):
    """Compare results between data sets for a list of plot types."""
    print '> SIMS: ',
    print simlist
    print '> Plots: ',
    print plottypes

    gap = 0.5  # gap for xticks
    xmax = None  # work out xmax

    plotdata = search_dirs(rootdir, simlist, plottypes)

    # iterate over all the plots we have data for
    for plot, sims in plotdata.items():
        count = 1  # reset sim counter
        nsims = len(sims)  # number sims to compare
        print '> Comparing ' + str(nsims) + ' plots for \'' + plot + '\''

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        artists = []  # save the artists for the legend
        labels = []  # save the labels for the legend

        # sort the data
        # data = sorted(data, key=lambda d: d.keys(''))
        sims = sorted(sims, key=lambda d: d['id'], reverse=False)

        # iterate over all the sim data sets in the list for this plot
        for sim in sims:
            # sim label and data
            label = sim['label']
            labels.append(label)
            data = sim['data']
            # plot color (cyclic)
            color = list(plt.rcParams['axes.prop_cycle'])[count-1]['color']
            # work out width
            if 'width' in data:
                width = data['width']
            # work out maximum xtick value
            xmax = nsims * len(data['x']) + count + len(data['x']) - nsims
            print ' ... ' + data['type'] + ' plot',
            print str(count) + '/' + str(nsims)

            # boxplots
            if data['type'] == 'box':
                pos = np.arange(count, xmax, nsims + 1)
                # lastdata = data['y'][1]
                bp = ax.boxplot(data['y'], positions=pos, notch=False,
                                widths=width,
                                showfliers=False,
                                patch_artist=True)
                cpplot.set_box_colors(bp, count-1)
                artists.append(bp["boxes"][0])
            # lineplots
            elif data['type'] == 'line':
                ax.plot(data['x'], data['y'],
                        color=color, marker='s', lw=2.0,
                        label=label)
            # barplots
            elif data['type'] == 'bar':
                xmax = max(data['x'])
                start = gap + width*(count-1)
                end = (gap + width*nsims) * xmax
                step = gap + width*nsims
                ind = np.arange(start, end, step)
                ax.bar(ind, data['y'], width, color=color, label=label)
            # histograms
            elif data['type'] == 'hist':
                ax.hist(data['x'], data['y'], normed=1, histtype='step',
                        cumulative=True, stacked=True, fill=True,
                        color=color)
            else:
                print 'Error: no type \'' + data['type'] + '\''
            # increment plot count
            count += 1

        # finish the histogram (so teh stacked hist doesn't cover ones below)

        # set a few more params
        if data['type'] == 'box':
            plt.xlim(0, xmax)
            # ax.set_ylim(top=15000)
            xticks = np.arange(gap + (width*nsims),
                               xmax,
                               count)
            ax.set_xticks(xticks)
            ax.set_xticklabels(data['x'])
        elif data['type'] == 'line':
            xticks = np.arange(min(data['x']), max(data['x'])+1, 1)
            ax.set_xticks(xticks)
        elif data['type'] == 'bar':
            # ax.set_ylim(top=15000)
            start = gap + (width*(nsims-1))/2
            end = (gap + width*nsims) * xmax + gap
            step = gap + width*nsims
            xticks = np.arange(start, end, step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(data['x'])
        # elif data['type'] == 'hist':
            # ax.set_xticks(np.arange(0, max(data['x']), 5.0))

        # legend
        for label in labels:
            label = r'\textbf{' + label + '}'  # make labels bold
        if artists:
            ax.legend(artists, labels, loc='best')
            # ax.set_xticks([1, 2, 3])
            # ax.set_xticklabels(['180', '300', '600'])
        else:
            if 'hops_prr' in plot:
                ax.legend(labels, loc='best')
            elif 'hops_rdc' in plot:
                ax.legend(labels, loc='lower right')
            elif 'join' in plot:
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
                                plot + '_' + str(simlist),  # filename
                                rootdir + '/',  # directory
                                xlabel=data['xlabel'],
                                ylabel=data['ylabel'])
