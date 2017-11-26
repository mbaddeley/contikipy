#!/usr/bin/env python2.7
from __future__ import division

import sys
import re  # regex
import os  # for makedir
import argparse  # command line arguments
import csv
import numpy as np  # number crunching
from scipy.stats.mstats import mode
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
                     '|\s+h:(?P<hops>\d+)'\
                     '|\s+m:(?P<mac>\d+))+.*?$'
    node_re = re.compile(log_pattern + debug_pattern +
                         '(?P<rank>\d+), (?P<degree>\d+)')
    app_re = re.compile(log_pattern + debug_pattern +
                        '(?P<status>(TX|RX))\s+(?P<typ>\S+)' +
                        packet_pattern)
    sdn_re = re.compile(log_pattern + debug_pattern +
                        '(?P<status>(OUT|BUF|IN))\s+(?P<typ>\S+)' +
                        packet_pattern)
    join_re = re.compile(log_pattern + sdn_ctrl_pattern +
                         '\s*[c:]{0,}(?P<join>\d*)')
    pow_re = re.compile(log_pattern + '.*P \d+.\d+ (?P<seqid>\d+).*'
                        '\(radio (?P<all_radio>\d+\W{1,2}\d+).*'
                        '(?P<radio>\d+\W{1,2}\d+).*'
                        '(?P<all_tx>\d+\W{1,2}\d+).*'
                        '(?P<tx>\d+\W{1,2}\d+).*'
                        '(?P<all_listen>\d+\W{1,2}\d+).*'
                        '(?P<listen>\d+\W{1,2}\d+)')

    print '**** Creating log files in \'' + out + '\''
    app_log = parse(log, out + "log_app_traffic.log", app_re)
    sdn_log = parse(log, out + "log_sdn_traffic.log", sdn_re)
    join_log = parse(log, out + "log_join.log", join_re)
    pow_log = parse(log, out + "log_pow.log", pow_re)
    node_log = parse(log, out + "log_node.log", node_re)

    print '**** Read logs into dataframes...'
    node_df = None
    app_df = None
    pow_df = None
    sdn_df = None
    join_df = None

    # General DFs
    if(os.path.getsize(node_log.name) != 0):
        node_df = read_node(csv_to_df(node_log))
    if(os.path.getsize(app_log.name) != 0):
        app_df = read_app(csv_to_df(app_log))
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
         join_df=join_df)

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
def compare_results(rootdir, simlist, plotlist):
    print '**** Analyzing (comparing) results'
    print '> SIMS: ',
    print simlist
    print '> Plots: ',
    print plotlist
    index = []
    axes = []

    for root, dirs, files in os.walk(rootdir):
        for dir in dirs:
            if dir in simlist:
                print '> ... Scanning \"' + root + '/' + dir + '/\"'
                for f in os.listdir(os.path.join(root, dir)):
                    for plot in plotlist:
                        if f == (plot + '.pkl'):
                            print ' * found pickle!'
                            ax = pickle.load(file(os.path.join(root, dir, f)))
                            axes.append(ax)
    # compare results
    fig, ax = plt.subplots()
    for plot in axes:
        ax.plot(plot)


# ----------------------------------------------------------------------------#
def plot_prr_over_br(df):
    print '> Analysis: Plotting prr over bitrate (bar)'
    df = df.pivot_table(index=[df.index], columns=['hops'], values='prr')

    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Bitrate (s)')
    ax.set_ylabel('PRR \%')
    ax.legend(['1-Hop', '2-Hops', '3-Hops', '4-Hops', '5-Hops'], loc='best')
    return fig


# ----------------------------------------------------------------------------#
def plot_tr_over_br(df):
    print '> Analysis: Plotting traffic ratio over bitrate (bar)'
    df = df.pivot_table(index=[df.index], columns=['type'], values='ratio')

    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Bitrate (s)')
    ax.set_ylabel('Traffic ratio (\%)')
    ax.legend(list(df.columns.values), loc=1)
    return fig


# ----------------------------------------------------------------------------#
# General Plotting
# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#
def plot(plot_list, out, node_df=None, app_df=None, sdn_df=None, join_df=None):
    # try:
    for plot in plot_list:
        # General plots
        # hops vs rdc
        if plot == 'hops_rdc':
            df = node_df.groupby('hops')['rdc'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, out, x=df.index, y=df.rdc)
            set_fig_and_save(df, fig, ax, plot, out,
                             'Hops', 'Radio duty cycle (\%)')
        # hops vs prr
        elif plot == 'hops_prr':
            df = node_df.groupby('hops')['prr'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            fig, ax = plot_bar(df, plot, out, x=df.index, y=df.prr)
            set_fig_and_save(df, fig, ax, plot, out,
                             'Hops', 'PRR (\%)')
        # hops mean latency
        elif plot == 'hops_lat_mean':
            df = app_df[['hops', 'lat']].reset_index(drop=True)
            plot_line_mean(df, plot, out,
                           df.hops, 'hops',  'lat', 'Mean delay (ms)')
        # hops end to end latency
        elif plot == 'hops_lat_e2e':
            df = app_df[['lat', 'hops']].set_index('hops')
            df = df[np.isfinite(df['lat'])].reset_index()  # drop NaN rows
            df = df.pivot(index=df.index, columns='hops')['lat']
            plot_box(df, plot, out,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
        # flows end to end latency
        elif plot == 'flow_lat':
            df = app_df.pivot_table(index=app_df.groupby('app').cumcount(),
                                    columns=['app'],
                                    values='lat')
            # plot e2e delay of each flow
            plot_box(df, plot, out,
                     xlabel='Flow \#', ylabel='End-to-end delay (ms)')

        # SDN plots
        # histogram of join time
        elif plot == 'sdn_join':
            df = join_df.copy()
            # FIXME: time in seconds (use timedelta)
            df['time'] = join_df['time']/1000/1000
            plot_hist(df, 'join', out, 'join', 'Time (s)',
                      'time', 'Nodes joined (\#)')
        # traffic ratio
        elif plot == 'sdn_traffic_ratio':
            plot_tr(app_df, sdn_df, out)

    # except Exception as e:
    #         print 'Exception: ...'
    #         print e  # print the exception
    #         pass  # continue
            # sys.exit(0)


# ----------------------------------------------------------------------------#
def set_fig_and_save(df, fig, ax, desc, out,
                     xlabel='xlabel', ylabel='ylabel'):
    # set axis' labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # tight layout
    fig.set_tight_layout(False)
    # save for post analysis
    pickle.dump(ax, file(out + desc + '.pkl', 'w'))
    # save figure
    fig.savefig(out + 'fig_' + desc + '.pdf')
    # close all open figs
    plt.close('all')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_hist(df, desc, out, x, xlabel, y, ylabel):
    print '> Plotting ' + desc + ' (hist)'
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='hist', x=x, y=y, cumulative=True, ax=ax)
    ax.legend_.remove()
    fig, ax = set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_bar(df, desc, out, x, y, ylim=None):
    print '> Plotting ' + desc + ' (bar)'
    fig, ax = plt.subplots(figsize=(8, 6))

    ind = np.arange(len(x))
    width = 0.35       # the width of the bars

    ax.bar(x=ind, height=y, width=width)
    # bar2 = ax.bar(x=ind+width, height=y, width=width)

    # set xticks
    # ax.set_xticks(ind+(width/2))

    # set x-axis
    ax.set_xticks(np.arange(min(ind), max(ind)+1, 1.0))
    ax.set_xticklabels(x)
    # set y limits
    if ylim:
        ax.set_ylim(ylim)

    # ax.legend((bar1[0], bar2[0]), ('Men', 'Women'))

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_box(df, desc, out, x=None, xlabel='X', y=None, ylabel='Y'):
    print '> Plotting ' + desc + ' (box)'
    fig, ax = plt.subplots(figsize=(8, 6))
    # drop rows with all NaN
    df = df.dropna(how='all')
    # matplotlib needs a list
    data = np.column_stack(df.transpose().values.tolist())
    # Filter data using np.isnan
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.T, mask.T)]
    bp = ax.boxplot(filtered_data, patch_artist=True)

    # set the linewidth for all
    linewidth = 1.5
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=linewidth)
        # # change fill color
        box.set(facecolor='b')
    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=linewidth)
    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=linewidth)
    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=linewidth)
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='#e7298a', alpha=0.5)

    fig, ax = set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel)

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

    fig, ax = set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel)

    return fig


# ----------------------------------------------------------------------------#
def plot_line_mean(df, desc, out, x, xlabel, y, ylabel):
    print '> Plotting ' + desc + ' (line)'
    # get mean and errors
    gp = df.groupby(x)
    means = gp.mean()
    means.index = means.index.astype(int)
    errors = gp.std()
    # do plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(means.index, means.lat, errors.lat,
                linestyle='--', marker='s', linewidth=1.5, color='b',
                capsize=3)

    # set x-axis
    ax.set_xticks(np.arange(min(means.index), max(means.index)+1, 1.0))
    ax.set_xticklabels(x)
    # set y limits
    ax.set_ylim([min(errors['lat']), max(errors['lat'])+1])
    # legend
    ax.legend([ylabel], loc=2)

    fig, ax = set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel)

    return fig


# ----------------------------------------------------------------------------#
# SDN Plotting
# ----------------------------------------------------------------------------#
def plot_tr(app_df, sdn_df, out):
    # FIXME: We are only looking at FTQ/FTS
    sdn_cbr_len = (sdn_df['typ'] == 'NSU').sum()
    sdn_vbr_len = (sdn_df['typ'] == 'FTQ').sum() + \
                  (sdn_df['typ'] == 'FTS').sum()
    total = len(app_df) + sdn_cbr_len + sdn_vbr_len
    app_ratio = len(app_df)/total  # get app packets as a % of total
    sdn_cbr_ratio = sdn_cbr_len/total  # get sdn packets as a % of total
    sdn_vbr_ratio = sdn_vbr_len/total  # get sdn packets as a % of total
    df = pd.DataFrame([app_ratio, sdn_cbr_ratio, sdn_vbr_ratio],
                      index=['App', 'SDN-CBR', 'SDN-VBR'],
                      columns=['ratio'])
    df.index.name = 'type'
    fig, ax = plot_bar(df, 'traffic_ratio', out, x=df.index,
                       y=df.ratio, ylim=1)
    set_fig_and_save(df, fig, ax, 'traffic_ratio', out,
                     'Traffic type', 'Percentage of total traffic (\%)')

    return fig, ax


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
