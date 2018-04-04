#!/usr/bin/python
"""Parse logs and generate results.

This module parses cooja logs according to a list of required data.
"""
from __future__ import division

import os  # for makedir
import pickle
import re  # regex
import sys

import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats.mstats import mode

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
def parse_log(datatype, log, dir, fmt, regex):
    """Parse the main log for data."""
    # Print some information about what's being parsed
    info = 'Data: {0} | Log: {1} ' \
           '| Log Format: {2}'.format(datatype, log, fmt)
    print '-' * len(info)
    print info
    print '-' * len(info)

    # dictionary of the various data df formatters
    read_fnc_dict = {
        "pow":  read_pow,
        "app":  read_app,
        "sdn":  read_sdn,
        "node": read_node,
        "icmp": read_icmp
    }

    try:
        # check the simulation directory exists, and there is a log there
        open(log, 'rb')
        # do the parsing
        print '**** Parsing log using ' + datatype + ' regex....'
        data_re = re.compile(regex)
        data_log = parse(log, dir + "log_" + datatype + ".log", data_re)
        if (os.path.getsize(data_log.name) != 0):
            data_df = csv_to_df(data_log)
            if datatype in read_fnc_dict.keys():
                # do some formatting on the df
                data_df = read_fnc_dict[datatype](data_df)
            elif 'join' in datatype:
                data_df = data_df
            else:
                data_df = None

            if data_df is not None:
                return data_df
            else:
                raise Exception('ERROR: Dataframe was None!')
        else:
            print 'WARN: Log was empty'
    except Exception as e:
            print e
            sys.exit(0)


# ----------------------------------------------------------------------------#
def extract_data(df_dict):
    """Take the dfs generated from the main log and analyze."""
    print '**** Do some additional processing on the dataframes...'
    # get general node data
    node_df = df_dict.get('node')
    if node_df is not None:
        app_df = df_dict.get('app')
        if app_df is not None:
            node_df = add_mean_lat_to_node_df(node_df, app_df)
            node_df = add_prr_to_node_df(node_df, app_df)
            node_df = add_hops_to_node_df(node_df, app_df)
            # node_df = node_df[np.isfinite(node_df['hops'])]  # drop NaN rows
            # node_df.hops = node_df.hops.astype(int)  # convert to int

        pow_df = df_dict.get('pow')
        if pow_df is not None:
            node_df = add_rdc_to_node_df(node_df, pow_df)
        df_dict['node'] = node_df

    print '> Node data summary...'
    print df_dict['node']


# ----------------------------------------------------------------------------#
def pickle_data(dir, data):
    """Save data by pickling it."""
    print '**** Pickling DataFrames ...'
    print data.keys()
    for k, v in data.items():
        print '> Saving ' + k
        if v is not None:
            v.to_pickle(dir + k + '_df.pkl')


# ----------------------------------------------------------------------------#
def plot_data(dir, data, plots):
    """Plot data according to required plot types."""
    print '**** Generating plots...' + str(plots)
    node_df = data['node']
    app_df = data['app']
    icmp_df = data['icmp']
    join_df = data['join']
    if 'sdn' in data.keys():
        sdn_df = data['sdn']
    else:
        sdn_df = None
    plot(plots, dir, node_df=node_df, app_df=app_df, sdn_df=sdn_df,
         icmp_df=icmp_df, join_df=join_df)


# ----------------------------------------------------------------------------#
# Parse main log using regex
# ----------------------------------------------------------------------------#
def parse(file_from, file_to, pattern):
    """Parse a log using regex and save in new log."""
    # Let's us know this is the first line and we need to write a header.
    write_header = 1
    # open the files
    with open(file_from, 'rb') as f:
        with open(file_to, 'wb') as t:
            for l in f:
                # HACK: Fixes issue with '-' in pow
                m = pattern.match(l.replace('.-', '.'))
                if m:
                    g = m.groupdict('')
                    if write_header:
                        t.write(','.join(g.keys()))
                        t.write('\n')
                        write_header = 0
                    t.write(','.join(g.values()))
                    t.write('\n')
                continue
    # Remember to close the logs!
    f.close()
    t.close()

    return t


# ----------------------------------------------------------------------------#
# Read logs
# ----------------------------------------------------------------------------#
def csv_to_df(file):
    """Create df from csv."""
    df = pd.read_csv(file.name)
    # drop any ampty columns
    df = df.dropna(axis=1, how='all')
    return df


# ----------------------------------------------------------------------------#
def read_app(df):
    """Read log for application data."""
    print '> Read app log'
    # sort the table by src/dest/seq so txrx pairs will be next to each other
    # this fixes NaN hop counts being filled incorrectly
    df = df.sort_values(['src', 'dest', 'app', 'seq']).reset_index(drop=True)
    # Rearrange columns
    df = df[['id', 'status', 'src', 'dest', 'app', 'seq', 'time',
             'hops', 'typ', 'module', 'level']]
    # fill in hops where there is a TX/RX
    df['hops'] = df.groupby(['src', 'dest', 'app', 'seq'])['hops'].apply(
                            lambda x: x.fillna(x.mean()))
    # pivot the table so we combine tx and rx rows for the same (src/dest/seq)
    df = df.bfill().pivot_table(index=['src', 'dest', 'app', 'seq', 'hops'],
                                columns=['status'],
                                values='time') \
                   .reset_index() \
                   .rename(columns={'TX': 'txtime',
                                    'RX': 'rxtime'})
    # remove the columns' name
    df.columns.name = None
    # format column types
    df['hops'] = df['hops'].astype(int)
    # add a 'dropped' column
    df['drpd'] = df['rxtime'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['rxtime'] - df['txtime'])/1000  # FIXME: /1000 = ns -> ms

    return df


# ----------------------------------------------------------------------------#
def read_icmp(df):
    """Read log for icmp data."""
    print '> Read icmp log'
    # TODO: Possibly do some processing?
    # print (df['type'] == 155).sum()
    # print (df['type'] == 200).sum()
    return df


# ----------------------------------------------------------------------------#
def read_sdn(sdn_df):
    """Read log for sdn data."""
    print '> Read sdn log'

    # Rearrange columns
    sdn_df = sdn_df[['src', 'dest', 'typ', 'seq', 'time',
                     'status', 'id', 'hops']]
    # Fixes settingwithcopywarning
    df = sdn_df.copy()
    # Fill in empty hop values for tx packets
    df['hops'] = sdn_df.groupby(['src', 'dest']).ffill().bfill()['hops']
    # Pivot table. Lose the 'mac' and 'id' column.
    df = df.pivot_table(index=['src', 'dest', 'typ', 'seq', 'hops'],
                        columns=['status'],
                        aggfunc={'time': np.sum},
                        values=['time'],
                        )
    # Get rid of the multiindex and rename the columns
    # TODO: not very elegant but it does the job
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df.columns = ['src', 'dest', 'typ', 'seq',
                  'hops', 'buf_t', 'in_t', 'out_t']
    # Set dest typ to int
    df.dest = df.dest.astype(int)
    # add a 'dropped' column
    df['drpd'] = df['in_t'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['buf_t'] - df['out_t'])/1000  # FIXME: /1000 = ns -> ms

    return df


# ----------------------------------------------------------------------------#
def read_pow(df):
    """Read log for power data."""
    print '> Read power log'
    # get last row of each 'id' group and use the all_radio value
    df = df.groupby('id').last()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


# ----------------------------------------------------------------------------#
def read_node(df):
    """Read log for node data."""
    print '> Read log for node data'
    df = df.groupby('id')['rank', 'degree'].agg(lambda x: min(x.mode()))
    return df


# ----------------------------------------------------------------------------#
# Additional processing
# ----------------------------------------------------------------------------#
def prr(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100


# ----------------------------------------------------------------------------#
def add_prr_to_node_df(node_df, app_df):
    """Add prr for each node."""
    print '> Add prr for each node'
    node_df['prr'] = app_df.groupby('src')['drpd'].apply(
                     lambda x: prr(len(x), x.sum()))
    return node_df


# ----------------------------------------------------------------------------#
def add_rdc_to_node_df(node_df, pow_df):
    """Add rdc for each node."""
    print '> Add rdc for each node'
    node_df = node_df.join(pow_df['all_radio'].rename('rdc'))
    return node_df


# ----------------------------------------------------------------------------#
def add_mean_lat_to_node_df(node_df, app_df):
    """Add mean lat for each node."""
    print '> Add mean_lat for each node'
    node_df['mean_lat'] = app_df.groupby('src')['lat'].apply(
                          lambda x: x.mean())
    return node_df


# ----------------------------------------------------------------------------#
def add_hops_to_node_df(node_df, app_df):
    """Add hops for each node."""
    print '> Add hops for each node'
    # Add hops to node_df. N.B. cols with NaN are always converted to float
    hops = app_df[['src', 'hops']].groupby('src').agg(lambda x: mode(x)[0])
    node_df = node_df.join(hops['hops'].astype(int))

    return node_df
    # convert to HH:mm:ss:ms
    # app_df['rxtime'] = pd.to_timedelta(app_df.rxtime/10000, unit='ms')
    # app_df['txtime'] = pd.to_timedelta(app_df.txtime/10000, unit='ms')
    # get back to ms
    # df.time.dt.total_seconds() * 1000


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
# def compare_boxplots():
#     pos = np.arange(count, xmax, nsims + 1)
#     # lastdata = data['y'][1]
#     bp = ax.boxplot(data['y'], positions=pos, notch=True,
#                     widths=width,
#                     showfliers=False,
#                     patch_artist=True)
#     set_box_colors(bp, count-1)
#     artists.append(bp["boxes"][0])

# ----------------------------------------------------------------------------#
def compare_results(rootdir, simlist, plottypes, **kwargs):
    print '**** Analyzing (comparing) results'
    print '> SIMS: ',
    print simlist
    print '> Plots: ',
    print plottypes

    plotdata = {}  # dictionary of plot data
    gap = 0.5  # gap for xticks
    xmax = None  # work out xmax

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
                            plotdata[plot].append({dir: d})
                            found = True
                    if not found:
                        print '- None'
    # lastdata = []
    # iterate over all the plots we have data for
    for plot in plotdata:
        count = 1  # reset sim counter
        nsims = len(plotdata[plot])  # number sims to compare
        print '> Comparing ' + str(nsims) + ' plots for \'' + plot + '\''

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        artists = []  # save the artists for the legend
        labels = []  # save the labels for the legend

        # print plotdata[plot]

        # iterate over all the sim data sets in the list for this plot
        for sim in plotdata[plot]:
            # sim label and data
            label = sim.keys()[0]
            labels.append(label)
            data = sim.values()[0]
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
                bp = ax.boxplot(data['y'], positions=pos, notch=True,
                                widths=width,
                                showfliers=False,
                                patch_artist=True)
                set_box_colors(bp, count-1)
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
                print data['y']
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
            ax.legend(artists, labels, loc='upper left')
        else:
            if 'hops_prr' in plot:
                ax.legend(labels, loc='best')
            if 'hops_rdc' in plot:
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
        fig, ax = set_fig_and_save(fig, ax, None,
                                   plot + '_' + str(simlist),  # filename
                                   rootdir + '/',  # directory
                                   xlabel=data['xlabel'],
                                   ylabel=data['ylabel'])


# ----------------------------------------------------------------------------#
# General Plotting
# ----------------------------------------------------------------------------#
def plot(plot_list, dir, node_df=None, app_df=None, sdn_df=None,
         icmp_df=None, join_df=None):

    for plot in plot_list:
        # General plots
        # hops vs rdc
        if plot == 'hops_rdc':
            df = node_df.groupby('hops')['rdc'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, dir, df.index, df.rdc,
                               xlabel='Hops', ylabel='Radio duty cycle (\%)')
        # hops vs prr
        elif plot == 'hops_prr':
            df = node_df.groupby('hops')['prr'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, dir, df.index, df.prr,
                               xlabel='Hops', ylabel='PDR (\%)')
        # hops mean latency
        elif plot == 'hops_lat_mean':
            df = app_df[['hops', 'lat']].reset_index(drop=True)
            gp = df.groupby('hops')
            means = gp.mean()
            plot_line(df, plot, dir, means.index, means.lat,
                      xlabel='Hops',  ylabel='Mean delay (ms)', steps=1.0)
        # hops end to end latency
        elif plot == 'hops_lat_e2e':
            df = app_df.pivot_table(index=app_df.groupby('hops').cumcount(),
                                    columns=['hops'],
                                    values='lat')
            # drop rows with all NaN
            df = df.dropna(how='all')
            # matplotlib needs a list
            data = np.column_stack(df.transpose().values.tolist())
            # ticks are the column headers
            xticks = list(df.columns.values)
            plot_box(plot, dir, xticks, data,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
        # flows end to end latency
        elif plot == 'flow_lat':
            df = app_df.pivot_table(index=app_df.groupby('app').cumcount(),
                                    columns=['app'],
                                    values='lat')
            # drop rows with all NaN
            df = df.dropna(how='all')
            # matplotlib needs a list
            data = np.column_stack(df.transpose().values.tolist())
            # ticks are the column headers
            xticks = list(df.columns.values)
            plot_box(plot, dir, xticks, data,
                     xlabel='Flow \#', ylabel='End-to-end delay (ms)')

        # SDN plots
        # histogram of join time
        elif plot == 'sdn_join':
            df = join_df.copy()
            # FIXME: time in seconds (use timedelta)

            df['time'] = join_df['time']/1000/1000
            # merge 'node' col into 'id' col, where the value in id is 1
            df.loc[df.id == 1, 'id'] = df.node
            # drop the node/module/level columns
            df = df.drop('node', 1)
            df = df.drop('module', 1)
            df = df.drop('level', 1)
            # merge dis,dao,controller
            # df = df.set_index(['time', 'id']).stack().reset_index()
            df = (df.set_index(['time', 'id'])
                  .stack()
                  .reorder_levels([2, 0, 1])
                  .reset_index(name='a')
                  .drop('a', 1)
                  .rename(columns={'level_0': 'type'}))
            # pivot so we use the type column as our columns
            df = df.pivot_table(index=['id'],
                                columns=['type'],
                                values='time').dropna(how='any')
            plot_hist('c_join', dir,
                      df['controller'].tolist(), df.index.tolist(),
                      xlabel='Time (s)', ylabel='Propotion of Nodes Joined')
            # plot_hist('dao_join', dir,
            #           df['dao'].tolist(), df.index.tolist(),
            #           xlabel='Time (s)', ylabel='Propotion of Nodes Joined')
            plot_hist('dag_join', dir,
                      df['dag'].tolist(), df.index.tolist(),
                      xlabel='Time (s)', ylabel='Propotion of Nodes Joined')
        # traffic ratio
        elif plot == 'sdn_traffic_ratio':
            plot_tr(app_df, sdn_df, icmp_df, dir)


# ----------------------------------------------------------------------------#
# Actual graph plotting functions
# ----------------------------------------------------------------------------#
def set_fig_and_save(fig, ax, data, desc, dir, **kwargs):

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
        pickle.dump(data, file(dir + desc + '.pkl', 'w'))

    # save figure
    fig.savefig(dir + 'fig_' + desc + '.pdf')

    # close all open figs
    plt.close('all')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_hist(desc, dir, x, y, **kwargs):
    print '> Plotting ' + desc + ' (hist)'
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
def checkstring(obj):
        return all(isinstance(elem, basestring) for elem in obj)


# ----------------------------------------------------------------------------#
def plot_bar(df, desc, dir, x, y, ylim=None, **kwargs):
    print '> Plotting ' + desc + ' (bar)'
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
    if not checkstring(x):
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
    print '> Plotting ' + desc + ' (box)'

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
    filtered_data = [d[m] for d, m in zip(y.T, mask.T)]
    bp = ax.boxplot(filtered_data, showfliers=False, patch_artist=True)
    set_box_colors(bp, 0)

    data = {'x': x, 'y': filtered_data,
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
    print '> Plotting ' + desc + ' (violin)'
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
    print '> Plotting ' + desc + ' (line)'

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
# SDN Plotting
# ----------------------------------------------------------------------------#
def plot_tr(app_df, sdn_df, icmp_df, dir):
    # FIXME: We are only looking at FTQ/FTS
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


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
