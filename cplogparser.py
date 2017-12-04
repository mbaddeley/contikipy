#!/usr/bin/env python2.7
from __future__ import division

import argparse  # command line arguments
import os  # for makedir
import pickle
import re  # regex
import sys

import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
from scipy.stats.mstats import mode

global node_df

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
def generate_results(log, out, fmt, label, plots):
    global node_df

    # Print some information about what's being parsed
    info = 'Parsing directory: {0} | Log Format: {1}' \
           '| Label: {2}'.format(out, fmt, label)
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
    if(os.path.getsize(icmp_log.name) != 0):
        icmp_df = read_icmp(csv_to_df(icmp_log))
    if(os.path.getsize(pow_log.name) != 0):
        pow_df = read_pow(csv_to_df(pow_log))
    # Specific SDN dfs
    if 'SDN' in label:
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
    # print (df['type'] == 155).sum()
    # print (df['type'] == 200).sum()
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

    # iterate over all the plots we have data for
    for plot in plotdata:
        count = 1  # reset sim counter
        total = len(plotdata[plot])  # number sims to compare
        print '> Comparing ' + str(total) + ' plots for \'' + plot + '\''

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        artists = []  # save the artists for the legend
        labels = []  # save the labels for the legend

        # iterate over all the sim data sets in the list for this plot
        for sim in plotdata[plot]:
            print ' ... ' + str(count) + '/' + str(total)
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
            xmax = total * len(data['x']) + count + len(data['x']) - total
            # boxplots
            if data['type'] == 'box':
                pos = np.arange(count, xmax, total + 1)
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
                end = (gap + width*total) * xmax
                step = gap + width*total
                ind = np.arange(start, end, step)
                ax.bar(ind, data['y'], width, color=color, label=label)
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
            # ax.set_ylim(top=15000)
            xticks = np.arange(gap + (width*total),
                               xmax,
                               count)
            ax.set_xticks(xticks)
            ax.set_xticklabels(data['x'])
        elif data['type'] == 'line':
            xticks = np.arange(min(data['x']), max(data['x'])+1, 1)
            ax.set_xticks(xticks)
            ax.legend(labels, loc='upper left')
        elif data['type'] == 'bar':
            # ax.set_ylim(top=15000)
            start = gap + (width*(total-1))/2
            end = (gap + width*total) * xmax + gap
            step = gap + width*total
            xticks = np.arange(start, end, step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(data['x'])
        elif data['type'] == 'hist':
            ax.set_xticks(np.arange(0, max(data['x']), 5.0))

        # legend
        for label in labels:
            label = r'\textbf{' + label + '}'  # make labels bold
        if artists:
            ax.legend(artists, labels, loc='best')
        else:
            ax.legend(labels, loc='best')
        # save figure
        fig, ax = set_fig_and_save(fig, ax, None,
                                   plot + '_' + str(simlist),  # filename
                                   rootdir + '/',  # directory
                                   xlabel=data['xlabel'],
                                   ylabel=data['ylabel'])


# ----------------------------------------------------------------------------#
# General Plotting
# ----------------------------------------------------------------------------#
def plot(plot_list, out, node_df=None, app_df=None, sdn_df=None,
         icmp_df=None, join_df=None):

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
                               xlabel='Hops', ylabel='PDR (\%)')
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
                      xlabel='Time (s)', ylabel='Nodes Joined (\%)')
            plot_hist('r_join', out,
                      df['SDN-RPL'].tolist(), df.index.tolist(),
                      xlabel='Time (s)', ylabel='Nodes Joined (\%)')
        # traffic ratio
        elif plot == 'sdn_traffic_ratio':
            plot_tr(app_df, sdn_df, icmp_df, out)


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

    data = {'x': x, 'y': y,
            'type': 'violin',
            'xlabel': xlabel,
            'ylabel': ylabel}

    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_line(df, desc, out, x, y, **kwargs):
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
    fig, ax = set_fig_and_save(fig, ax, data, desc, out,
                               xlabel=xlabel, ylabel=ylabel)
    return fig, ax


# ----------------------------------------------------------------------------#
# SDN Plotting
# ----------------------------------------------------------------------------#
def plot_tr(app_df, sdn_df, icmp_df, out):
    # FIXME: We are only looking at FTQ/FTS
    if sdn_df is not None:
        sdn_cbr_len = (sdn_df['typ'] == 'NSU').sum()
        sdn_vbr_len = (sdn_df['typ'] == 'FTQ').sum() + \
                      (sdn_df['typ'] == 'FTS').sum()
        sdn_icmp_count = (icmp_df['type'] == 200).sum()

    rpl_icmp_count = (icmp_df['type'] == 155).sum()
    if sdn_df is not None:
        total = len(app_df) + sdn_cbr_len + sdn_vbr_len \
                + rpl_icmp_count + sdn_icmp_count
    else:
        total = len(app_df) + rpl_icmp_count

    app_ratio = len(app_df)/total  # get app packets as a % of total
    if sdn_df is not None:
        sdn_cbr_ratio = sdn_cbr_len/total  # get sdn packets as a % of total
        sdn_vbr_ratio = sdn_vbr_len/total  # get sdn packets as a % of total
        sdn_icmp_ratio = sdn_icmp_count/total
    rpl_icmp_ratio = rpl_icmp_count/total

    if sdn_df is not None:
        df = pd.DataFrame([app_ratio, rpl_icmp_ratio, sdn_icmp_ratio,
                           sdn_cbr_ratio, sdn_vbr_ratio],
                          index=['App', 'RPL', 'SDN-ICMP',
                                 'SDN-CBR', 'SDN-VBR'],
                          columns=['ratio'])
    else:
        df = pd.DataFrame([app_ratio, rpl_icmp_ratio],
                          index=['App', 'RPL'],
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
