#!/usr/bin/env python2.7
from __future__ import division

import sys
import re  # regex
import os  # for makedir
import argparse  # command line arguments
import csv
import numpy as np  # number crunching
from scipy.stats.mstats import mode
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt  # general plotting
import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
import platform
import yaml
import pickle

global node_df

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)
plt.style.use('seaborn-deep')

# Pandas options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ----------------------------------------------------------------------------#
def main():
    """ Main function is for standalone operation """
    # Fetch arguments
    ap = argparse.ArgumentParser(description='Simulation log parser')
    ap.add_argument('--log', required=True,
                    help='Absolute path to log file')
    ap.add_argument('--out', required=False,
                    default="./",
                    help='Where to save logs and plots')
    ap.add_argument('--fmt', required=False, default="cooja",
                    help='Format of file to parse')
    ap.add_argument('--desc', required=False,
                    default="",
                    help='Simulation desc to prefix to output graphs')
    args = ap.parse_args()

    # Generate the results
    generate_results(args.log, args.out, args.fmt, args.desc)


# ----------------------------------------------------------------------------#
def generate_results(log, out, fmt, desc, plots):
    global node_df

    # Print some information about what's being parsed
    info = 'Parsing directory: {0} | Log Format: {1}'.format(out, fmt)
    print '-' * len(info)
    print info
    print '-' * len(info)

    # check the simulation directory exists, and there is a log there
    try:
        open(log, 'rb')
    except Exception as e:
            print e
            sys.exit(0)

    # create a summary of the scenario
    summary_log = open(out + "log_summary.log", 'a')
    summary_log.write("Scenario\ttotal\tdropped\tprr\tduty_cycle\n")

    print '**** Compiling regex....'
    # log pattern
    log_pattern = {'cooja': '^\s*(?P<time>\d+):\s*(?P<node>\d+):',
                   }.get(fmt, None)
    if log_pattern is None:
        sys.stderr.write("Unknown record format: %s\n" % fmt)
        sys.exit(1)
    # debug pattern
    debug_pattern = '\s*(?P<module>[\w,-]+):\s*(?P<level>STAT):\s*'
    sdn_ctrl_pattern = '\s*(?P<module>SDN-CTRL):\s*(?P<level>STAT):\s*'
    # packet patterns ... https://regex101.com/r/mE5wK0/1
    packet_pattern = '(?:\s+s:(?P<src>\d+)'\
                     '|\s+d:(?P<dest>\d+)'\
                     '|\s+a:(?P<app>\d+)'\
                     '|\s+id:(?P<seq>\d+)'\
                     '|\s+h:(?P<hops>[1-5])'\
                     '|\s+m:(?P<mac>\d+))+.*?$'
    node_re = re.compile(log_pattern + debug_pattern +
                         '(?P<rank>\d+), (?P<degree>\d+)')
    app_re = re.compile(log_pattern + debug_pattern +
                        '(?P<status>(TX|RX))\s+(?P<typ>\S+)' +
                        packet_pattern)
    sdn_re = re.compile(log_pattern + debug_pattern +
                        '(?P<status>(OUT|BUF|IN))\s+(?P<typ>\S+)' +
                        packet_pattern)
    icmp_re = re.compile(log_pattern + debug_pattern +
                         '(?:\s+type:(?P<type>\d+)'
                         '|\s+code:(?P<code>\d+))+.*?$')
    join_re = re.compile(log_pattern + debug_pattern +
                         '(?:\s+c:(?P<controller>\d+)'
                         '|\s+r:(?P<dag>\d+))+.*?$')
    pow_re = re.compile(log_pattern + '.*P \d+.\d+ (?P<seqid>\d+).*'
                        '\(radio (?P<all_radio>\d+\W{1,2}\d+).*'
                        '(?P<radio>\d+\W{1,2}\d+).*'
                        '(?P<all_tx>\d+\W{1,2}\d+).*'
                        '(?P<tx>\d+\W{1,2}\d+).*'
                        '(?P<all_listen>\d+\W{1,2}\d+).*'
                        '(?P<listen>\d+\W{1,2}\d+)')

    print '**** Creating log files in \'' + out + '\''
    app_log = parse(log, out + "log_app_traffic.log", app_re)
    icmp_log = parse(log, out + "log_icmp.log", icmp_re)
    sdn_log = parse(log, out + "log_sdn_traffic.log", sdn_re)
    join_log = parse(log, out + "log_join.log", join_re)
    pow_log = parse(log, out + "log_pow.log", pow_re)
    node_log = parse(log, out + "log_node.log", node_re)

    print '**** Read logs into dataframes...'
    node_df = None
    app_df = None
    icmp_df = None
    pow_df = None
    sdn_df = None
    join_df = None

    # General DFs
    if(os.path.getsize(node_log.name) != 0):
        node_df = read_node(csv_to_df(node_log))
    if(os.path.getsize(app_log.name) != 0):
        app_df = read_app(csv_to_df(app_log))
    if(os.path.getsize(node_log.name) != 0):
        icmp_df = read_icmp(csv_to_df(icmp_log))
    if(os.path.getsize(pow_log.name) != 0):
        pow_df = read_pow(csv_to_df(pow_log))
    # Specific SDN dfs
    if 'SDN' in desc:
        if(os.path.getsize(sdn_log.name) != 0):
            sdn_df = read_sdn(csv_to_df(sdn_log))
        if(os.path.getsize(join_log.name) != 0):
            join_df = csv_to_df(join_log)

    print '**** Do some additional processing...'
    if app_df is not None and node_df is not None:
        node_df = add_mean_lat_to_node_df(node_df, app_df)
        node_df = add_prr_to_node_df(node_df, app_df)
    if pow_df is not None and node_df is not None:
        node_df = add_rdc_to_node_df(node_df, pow_df)

    if node_df is not None:
        # a bit of preparation
        node_df = node_df[np.isfinite(node_df['hops'])]  # drop NaN rows
        node_df.hops = node_df.hops.astype(int)

    prefix = out + desc + '_'

    print '**** Generating plots...' + str(plots)
    plot(plots, out, node_df=node_df, app_df=app_df, sdn_df=sdn_df,
         icmp_df=icmp_df, join_df=join_df)

    print '**** Pickling DataFrames ...'
    # general
    if node_df is not None:
        node_df.to_pickle(out + 'node_df.pkl')
    if app_df is not None:
        app_df.to_pickle(out + 'app_df.pkl')
    # sdn
    if sdn_df is not None:
        sdn_df.to_pickle(out + 'sdn_df.pkl')


# ----------------------------------------------------------------------------#
# Write logs
# ----------------------------------------------------------------------------#
def parse(file_from, file_to, pattern):
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
    df = pd.read_csv(file.name)
    # drop any ampty columns
    df = df.dropna(axis=1, how='all')

    return df


# ----------------------------------------------------------------------------#
def read_app(df):
    print '> Read app log'
    global node_df
    # sort the table by src/dest/seq so txrx pairs will be next to each other
    # this fixes NaN hop counts being filled incorrectly
    df = df.sort_values(['src', 'dest', 'app', 'seq']).reset_index(drop=True)
    # Rearrange columns
    df = df[['node', 'status', 'src', 'dest', 'app', 'seq', 'time',
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
    if node_df is not None:
        # Add hops to node_df. N.B. cols with NaN are always converted to float
        hops = df[['src', 'hops']].groupby('src').agg(lambda x: mode(x)[0])
        node_df = node_df.join(hops['hops'].astype(int))
        # convert to HH:mm:ss:ms
        # app_df['rxtime'] = pd.to_timedelta(app_df.rxtime/10000, unit='ms')
        # app_df['txtime'] = pd.to_timedelta(app_df.txtime/10000, unit='ms')
        # get back to ms
        # df.time.dt.total_seconds() * 1000
    return df

# ----------------------------------------------------------------------------#
def read_icmp(df):
    print '> Read icmp log'

    # TODO: Possibly do some processing?
    print (df['type'] == 155).sum()
    print (df['type'] == 200).sum()

    return df


# ----------------------------------------------------------------------------#
def read_sdn(sdn_df):
    print '> Read sdn log'
    global node_df

    # Rearrange columns
    sdn_df = sdn_df[['src', 'dest', 'typ', 'seq', 'time',
                     'status', 'node', 'hops']]
    # Fixes settingwithcopywarning
    df = sdn_df.copy()
    # Fill in empty hop values for tx packets
    df['hops'] = sdn_df.groupby(['src', 'dest']).ffill().bfill()
    # Pivot table. Lose the 'mac' and 'node' column.
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
    print '> Read power log'
    # get last row of each 'node' group and use the all_radio value
    df = df.groupby('node').last()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


# ----------------------------------------------------------------------------#
def read_node(df):
    print '> Read node log'
    df = df.groupby('node')['rank', 'degree'].agg(lambda x: min(x.mode()))
    return df


# ----------------------------------------------------------------------------#
# Additional processing
# ----------------------------------------------------------------------------#
def prr(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100


# ----------------------------------------------------------------------------#
def add_prr_to_node_df(node_df, app_df):
    print '> Add prr for each node'
    node_df['prr'] = app_df.groupby('src')['drpd'].apply(
                     lambda x: prr(len(x), x.sum()))
    return node_df


# ----------------------------------------------------------------------------#
def add_rdc_to_node_df(node_df, pow_df):
    print '> Add rdc for each node'
    node_df = node_df.join(pow_df['all_radio'].rename('rdc'))
    return node_df


# ----------------------------------------------------------------------------#
def add_mean_lat_to_node_df(node_df, app_df):
    print '> Add mean_lat for each node'
    node_df['mean_lat'] = app_df.groupby('src')['lat'].apply(
                          lambda x: x.mean())
    return node_df


# ----------------------------------------------------------------------------#
# Results analysis
# ----------------------------------------------------------------------------#
def set_box_colors(bp, index):
    color = list(plt.rcParams['axes.prop_cycle'])[index]['color']
    lw = 1.5
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
def compare_results(rootdir, simlist, plotlist, **kwargs):
    print '**** Analyzing (comparing) results'
    print '> SIMS: ',
    print simlist
    print '> Plots: ',
    print plotlist
    index = []
    datalist = {}
    labels = []

    # get kwargs
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    # iterate through root directory looking for pickles
    for plot in plotlist:
        # create new list in dict
        datalist[plot] = []
        for root, dirs, files in os.walk(rootdir):
            for dir in sorted(dirs):
                if dir in simlist:
                    print '> ... Scanning \"' + root + '/' + dir + '/\"'
                    for f in os.listdir(os.path.join(root, dir)):
                        if (plot + '.pkl') in f:
                            print ' * found pickle!'
                            d = pickle.load(file(os.path.join(root, dir, f)))
                            datalist[plot].append({dir: d})

    # count number of data sets
    for plot in plotlist:
        # create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # generate a description for this plot
        description = plot + '_' + str(simlist)
        # number of plots we are analyzing
        num_plots = len(datalist[plot])
        # keep track of which plot we are on
        count = 1
        # save the artists for the legend
        artists = []
        # work out xmax
        xmax = 0
        print '> Comparing ' + str(num_plots) + ' plots for ' + plot
        for sim_data in datalist[plot]:
            sim = sim_data.keys()[0]
            data = sim_data.values()[0]
            color = list(plt.rcParams['axes.prop_cycle'])[count-1]['color']
            # boxplots
            if data['type'] == 'box':
                xmax = num_plots * len(data['x']) + \
                       (count - 1) + len(data['x']) - num_plots + 1
                pos = np.arange(count, xmax, num_plots + 1)
                print pos
                bp = ax.boxplot(data['y'], positions=pos, notch=True,
                                widths=data['width'],
                                patch_artist=True)
                set_box_colors(bp, count-1)
                artists.append(bp["boxes"][0])
            # lineplots
            elif data['type'] == 'line':
                ax.plot(data['x'], data['y'],
                        color=color, marker='s', lw=2.0,
                        label=sim)
            # barplots
            elif data['type'] == 'bar':
                ax.bar((data['x'] + data['width']*count), data['y'],
                       width=data['width'],
                       color=color, label=sim)
            # histograms
            elif data['type'] == 'hist':
                ax.hist(data['x'], len(data['y']), normed=1, histtype='step',
                        cumulative=True, stacked=True, fill=True,
                        color=color)
            else:
                print 'Error: no type \'' + data['type'] + '\''
            # increment plot count
            count += 1
        # set a few more params
        if data['type'] == 'box':
            plt.xlim(0, xmax)
            xticks = np.arange(1.5,
                               xmax,
                               count)
            ax.set_xticks(xticks)
            ax.set_xticklabels(data['x'])
            ax.legend(artists, simlist, loc='upper right')
        elif data['type'] == 'line':
            xticks = np.arange(min(data['x']), max(data['x'])+1, 1)
            ax.set_xticks(xticks)
            ax.legend(loc='upper left')
        elif data['type'] == 'bar':
            ax.set_xticks(data['x'] + data['width'] +
                          (data['width']/count) + 0.05)
            ax.legend(loc='upper right')
            ax.set_xticklabels(data['x'])
        elif data['type'] == 'hist':
            ax.set_xticks(np.arange(0, max(data['x']), 5.0))
            ax.legend(['RPL', 'SDN'], loc='upper left')

        # save figure
        fig, ax = set_fig_and_save(fig, ax, None, description, rootdir + '/',
                                   xlabel=data['xlabel'],
                                   ylabel=data['ylabel'])


# ----------------------------------------------------------------------------#
def plot_prr_over_br(df):
    print '> Analysis: Plotting prr over bitrate (bar)'
    df = df.pivot_table(index=[df.index], columns=['hops'], values='prr')

    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Bitrate (s)')
    ax.set_ylabel('PRR \%')
    ax.legend(['1-Hop', '2-Hops', '3-Hops', '4-Hops', '5-Hops'], loc='best')
    return fig, ax


# ----------------------------------------------------------------------------#
def plot_tr_over_br(df):
    print '> Analysis: Plotting traffic ratio over bitrate (bar)'
    df = df.pivot_table(index=[df.index], columns=['type'], values='ratio')

    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Bitrate (s)')
    ax.set_ylabel('Traffic ratio (\%)')
    ax.legend(list(df.columns.values), loc=1)
    return fig, ax


# ----------------------------------------------------------------------------#
# General Plotting
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#
def plot(plot_list, out, node_df=None, app_df=None, sdn_df=None,
         icmp_df=None, join_df=None):
    # try:
    for plot in plot_list:
        # General plots
        # hops vs rdc
        if plot == 'hops_rdc':
            df = node_df.groupby('hops')['rdc'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, out, df.index, df.rdc,
                               xlabel='Hops', ylabel='Radio duty cycle (\%)')
        # hops vs prr
        elif plot == 'hops_prr':
            df = node_df.groupby('hops')['prr'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, out, df.index, df.prr,
                               xlabel='Hops', ylabel='PRR (\%)')
        # hops mean latency
        elif plot == 'hops_lat_mean':
            df = app_df[['hops', 'lat']].reset_index(drop=True)
            gp = df.groupby('hops')
            means = gp.mean()
            plot_line(df, plot, out, means.index, means.lat,
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
            plot_box(plot, out, xticks, data,
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
            plot_box(plot, out, xticks, data,
                     xlabel='Flow \#', ylabel='End-to-end delay (ms)')

        # SDN plots
        # histogram of join time
        elif plot == 'sdn_join':
            df = join_df.copy()
            # FIXME: time in seconds (use timedelta)
            df['time'] = join_df['time']/1000/1000
            df = df.pivot_table(index=['node'],
                                columns=['module'],
                                values='time').dropna(how='any')
            plot_hist('c_join', out,
                      df['SDN-CTRL'].tolist(), df.index.tolist(),
                      xlabel='Time (s)', ylabel='Nodes joined (\%)')
            plot_hist('r_join', out,
                      df['SDN-RPL'].tolist(), df.index.tolist(),
                      xlabel='Time (s)', ylabel='Nodes joined (\%)')
        # traffic ratio
        elif plot == 'sdn_traffic_ratio':
            plot_tr(app_df, sdn_df, icmp_df, out)

    # except Exception as e:
    #         print 'Exception: ...'
    #         print e  # print the exception
    #         pass  # continue
            # sys.exit(0)


# ----------------------------------------------------------------------------#
def set_fig_and_save(fig, ax, data, desc, out, **kwargs):

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
        pickle.dump(data, file(out + desc + '.pkl', 'w'))

    # save figure
    fig.savefig(out + 'fig_' + desc + '.pdf')

    # close all open figs
    plt.close('all')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_hist(desc, out, x, y, **kwargs):
    print '> Plotting ' + desc + ' (hist)'
    fig, ax = plt.subplots(figsize=(8, 6))

    # get kwargs
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    ax.hist(x, len(y), normed=1, histtype='step', cumulative=True,
            stacked=True, fill=True, label=desc)
    ax.set_xticks(np.arange(0, max(x), 5.0))
    # ax.legend_.remove()

    data = {'x': x, 'y': y,
            'type': 'hist',
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_bar(df, desc, out, x, y, ylim=None, **kwargs):
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
    ax.set_xticklabels(x)
    # set y limits
    if ylim is not None:
        ax.set_ylim(ylim)

    # ax.legend((bar1[0], bar2[0]), ('Men', 'Women'))

    data = {'x': x, 'y': y,
            'type': 'bar',
            'width': width,
            'xlabel': xlabel,
            'ylabel': ylabel}
    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_box(desc, out, x, y, **kwargs):
    print '> Plotting ' + desc + ' (box)'

    # subfigures
    fig, ax = plt.subplots(figsize=(8, 6))

    # constants
    # ylim = [0, 1500]
    width = 0.5   # the width of the boxes
    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # get kwargs
    steps = kwargs['steps'] if 'steps' in kwargs else 1
    color = kwargs['color'] if 'color' in kwargs else color
    ylim = kwargs['ylim'] if 'ylim' in kwargs else None
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else ''
    ylabel = kwargs['ylabel'] if 'ylabel' in kwargs else ''

    # set xticks
    xticks = np.arange(min(x), max(x)+1, steps)

    # Filter data using np.isnan
    mask = ~np.isnan(y)
    filtered_data = [d[m] for d, m in zip(y.T, mask.T)]
    bp = ax.boxplot(filtered_data, patch_artist=True)
    set_box_colors(bp, 0)

    data = {'x': x, 'y': filtered_data,
            'type': 'box',
            'width': width,
            'xlabel': xlabel,
            'ylabel': ylabel}

    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               ylim=ylim,
                               xlabel=xlabel,
                               ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_violin(df, desc, out, x, xlabel, y, ylabel):
    print '> Plotting ' + desc + ' (violin)'
    fig, ax = plt.subplots(figsize=(8, 6))

    xticks = [0, 1, 2, 3, 4, 5, 6]
    ax.xaxis.set_ticks(xticks)

    ax.violinplot(dataset=[df[df[x] == 1][y],
                           df[df[x] == 2][y],
                           df[df[x] == 3][y],
                           df[df[x] == 4][y]])

    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_line(df, desc, out, x, y, **kwargs):
    print '> Plotting ' + desc + ' (line)'

    # constants
    capsize = 3
    lw = 2.0
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
    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)
    return fig, ax


# ----------------------------------------------------------------------------#
# SDN Plotting
# ----------------------------------------------------------------------------#
def plot_tr(app_df, sdn_df, icmp_df, out):
    # FIXME: We are only looking at FTQ/FTS
    sdn_cbr_len = (sdn_df['typ'] == 'NSU').sum()
    sdn_vbr_len = (sdn_df['typ'] == 'FTQ').sum() + \
                  (sdn_df['typ'] == 'FTS').sum()
    rpl_icmp_count = (icmp_df['type'] == 155).sum()
    sdn_icmp_count = (icmp_df['type'] == 200).sum()
    total = len(app_df) + sdn_cbr_len + sdn_vbr_len + rpl_icmp_count + sdn_icmp_count
    app_ratio = len(app_df)/total  # get app packets as a % of total
    sdn_cbr_ratio = sdn_cbr_len/total  # get sdn packets as a % of total
    sdn_vbr_ratio = sdn_vbr_len/total  # get sdn packets as a % of total
    rpl_icmp_ratio = rpl_icmp_count/total
    sdn_icmp_ratio = sdn_icmp_count/total
    df = pd.DataFrame([app_ratio, rpl_icmp_ratio, sdn_icmp_ratio, sdn_cbr_ratio, sdn_vbr_ratio],
                      index=['App', 'RPL', 'SDN-ICMP', 'SDN-CBR', 'SDN-VBR'],
                      columns=['ratio'])
    df.index.name = 'type'
    fig, ax = plot_bar(df, 'traffic_ratio', out, x=df.index,
                       y=df.ratio)
    data = {'x': df.index, 'y': df.ratio}
    set_fig_and_save(fig, ax, data, 'traffic_ratio', out,
                     xlabel='Traffic type',
                     ylabel='Percentage of total traffic (\%)')

    return fig, ax


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
