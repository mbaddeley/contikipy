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

global node_df

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)
# Pandas options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Results config
STD_CONFIG = {'RDC_HOPS': 1,
              'PRR_HOPS': 1,
              'MEAN_LAT_HOPS': 1,
              'E2E_LAT_HOPS': 1,
              'SDN_JOIN': 1,
              'SDN_TR': 1,
              'SCEN_RR': 0,
              }

# SDN reroute scenario config
SCEN_RR_CONFIG = {'RDC_HOPS': 0,
                  'PRR_HOPS': 0,
                  'MEAN_LAT_HOPS': 0,
                  'E2E_LAT_HOPS': 0,
                  'SDN_JOIN': 1,
                  'SDN_TR': 1,
                  'SCEN_RR': 1,
                  }

# Analysis config
A_CONFIG = {'TR_BR': 1,
            }


# ----------------------------------------------------------------------------#
def main():
    """ Main function is for standalone operation """
    # Fetch arguments
    ap = argparse.ArgumentParser(description='Simulation log parser')
    ap.add_argument('--log', required=False,
                    default='/home/mike/Results/COOJA_results.log',
                    # default='/home/mike/Repos/usdn/'
                            # 'tools/cooja/build/COOJA.testlog',
                    help='Log file to parse')
    ap.add_argument('--out', required=False,
                    default="/home/mike/Results/",
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
def generate_results(log, out, fmt, desc):
    global node_df

    config = SCEN_RR_CONFIG

    # Set some defaults
    if log is None:
        log = '/home/mike/Repos/usdn/tools/cooja/build/COOJA.testlog'
    if out is None:
        out = '/home/mike/Results/'
    if fmt is None:
        fmt = 'cooja'

    print ' log is ... ' + log

    # Summary log needs to be in the parent directory. If we are creating the
    # log then we need to set the column names
    summary_log = open(out + "log_summary.log", 'a')
    if (os.stat(summary_log.name).st_size == 0):
        summary_log.write("Scenario\ttotal\tdropped\tprr\tduty_cycle\n")

    # Print some information about what's being parsed
    info = 'Parsing scenario: {0} Dir:{1} Fmt: {2}'.format(
                 desc, out, fmt)
    print '*' * len(info)
    print info
    print '*' * len(info)

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
    node_df = read_node(csv_to_df(node_log))
    app_df = read_app(csv_to_df(app_log))
    pow_df = read_pow(csv_to_df(pow_log))
    if 'SDN_1' in desc:
        # SDN Specific
        sdn_df = read_sdn(csv_to_df(sdn_log))
        join_df = csv_to_df(join_log)

    print '**** Do some additional processing...'
    node_df = add_to_node_df(node_df, app_df, pow_df)
    summarize(desc, summary_log, app_df, pow_df, node_df)

    print '**** Plotting with matplotlib and saving figures...'
    # a bit of preparation
    node_df = node_df[np.isfinite(node_df['hops'])]  # drop NaN rows from node
    node_df.hops = node_df.hops.astype(int)

    prefix = out + desc + '_'

    # general plots
    # hops vs rdc
    if config['RDC_HOPS']:
        df = node_df[['rdc', 'hops']].set_index('hops')  # set idx + drop cols
        df = df[np.isfinite(df['rdc'])]  # drop NaN rows
        plot_bar(df, 'rdc_hops', out, df.index, 'Hops',
                 df.rdc, 'Radio duty cycle (\%)')
    # hops vs prr
    if config['PRR_HOPS']:
        df = node_df[['prr', 'hops']].set_index('hops')  # set idx + drop cols
        df = df[np.isfinite(df['prr'])]  # drop NaN rows
        plot_bar(df, 'prr_hops', out,  df.index, 'Hops',  df.prr, 'PRR (\%)')
    # mean latency
    if config['MEAN_LAT_HOPS']:
        df = app_df[['hops', 'lat']].reset_index(drop=True)
        plot_line_mean(df, 'mean_lat', out,
                       df.hops, 'Hops',  df.lat, 'Mean delay (ms)')
    # end to end latency
    if config['E2E_LAT_HOPS']:
        df = app_df[['lat', 'hops']].set_index('hops')
        df = df[np.isfinite(df['lat'])].reset_index()  # drop NaN rows
        df = df.pivot(index=df.index, columns='hops')['lat']
        plot_box(df, 'e2e_lat', out,  df.columns, 'Hops',
                 df.index, 'End-to-end delay (ms)')

    # sdn plots
    if 'SDN_1' in desc:
        # histogram of join time
        if config['SDN_JOIN']:
            df = join_df.copy()
            # FIXME: time in seconds (use timedelta)
            df['time'] = join_df['time']/1000/1000
            plot_hist(df, 'join', out, 'join', 'Time (s)',
                      'time', 'Nodes joined (\#)')
        # traffic ratio
        if config['SDN_TR']:
            plot_tr(app_df, sdn_df, out)

    # scenario plots
    # application scenario
    if config['SCEN_RR']:
        df = app_df[['app', 'lat']].set_index('app')
        df = df[np.isfinite(df['lat'])].reset_index()  # drop NaN rows
        df = df.pivot(index=df.index, columns='app')['lat']
        plot_reroute_scenario(df, out)

    # save dfs for later
    print '**** Pickling DataFrames ...'
    node_df.to_pickle(out + 'node_df.pkl')
    app_df.to_pickle(out + 'app_df.pkl')
    if 'SDN_1' in desc:
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
    # pivot the table so we combine tx and rx rows for the same (src/dest/seq)
    df = df.bfill().pivot_table(index=['src', 'dest', 'app', 'seq', 'hops'],
                                columns=['status'],
                                values='time'
                                ).reset_index().rename(
                                columns={'TX': 'txtime',
                                         'RX': 'rxtime'})
    # remove the columns' name
    df.columns.name = None
    # add a 'dropped' column
    df['drpd'] = df['rxtime'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['rxtime'] - df['txtime'])/1000  # FIXME: /1000 = ns -> ms
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
    df = df.groupby('node')['rank', 'degree'].agg(lambda x: x.mode())
    return df


# ----------------------------------------------------------------------------#
def summarize(desc, log, app_df, pow_df, node_df):
    print '> Summarizing scenario and saving to log'
    # PRR
    dropcount = np.count_nonzero(app_df['drpd'])
    totalcount = len(app_df['drpd'])
    recv_ratio = prr(totalcount, dropcount)
    if not desc:
        desc = 'Scenario_None'
    log.write('{0}\t{1}\t{2}\t{3}\t'.format(desc, totalcount,
                                            dropcount,
                                            round(recv_ratio, 2)))
    # RDC
    # print pow_df
    avg_rdc = pow_df['all_radio'].mean()
    log.write('{0}\n'.format(round(avg_rdc, 2)))


# ----------------------------------------------------------------------------#
# Additional processing
# ----------------------------------------------------------------------------#
def prr(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100


# ----------------------------------------------------------------------------#
def add_to_node_df(node_df, app_df, pow_df):
    print '> Add mean lat, prr, rdc for each node'
    # mean latency for node
    node_df['mean_lat'] = app_df.groupby('src')['lat'].apply(
                          lambda x: x.mean())
    # prr for node
    node_df['prr'] = app_df.groupby('src')['drpd'].apply(
                     lambda x: prr(len(x), x.sum()))
    # rdc for node
    node_df = node_df.join(pow_df['all_radio'].rename('rdc'))
    # sdn overhead for node
    # TODO
    return node_df


# ----------------------------------------------------------------------------#
# Results analysis
# ----------------------------------------------------------------------------#
def analyze_results():
    print '**** Analyzing Results'
    if A_CONFIG['PRR_BR']:
        # prr over br
        analyze_prr_over_br(args.out)
    if A_CONFIG['TR_BR']:
        # tr over br
        analyze_tr_over_br(args.out)


# ----------------------------------------------------------------------------#
def analyze_prr_over_br(outdir):
    print '**** Analyzing PRR over Application BR'
    df = plot_over(outdir, 'CBR|VBR', '.*SDN_0', 'prr')
    fig = plot_prr_over_br(df)
    fig.savefig(outdir + '/prr_over_br.pdf')


# ----------------------------------------------------------------------------#
def analyze_tr_over_br(outdir):
    print '**** Analyzing Traffic Ratio over Application BR'
    df = plot_over(outdir, 'CBR|VBR', '.*SDN_1', 'traffic_ratio')
    fig = plot_tr_over_br(df)
    fig.savefig(outdir + '/tr_over_br.pdf')


# ----------------------------------------------------------------------------#
def plot_over(folder, index_re, filter_re, pkl):
    ''' Searches for folders matching filter_re, and takes index
        using index_re '''
    index = []
    df_list = []
    regex = re.compile('^.*(?P<idx>' + index_re + ')_'
                       '(?P<value>[\d,_]+)_' + filter_re + '.*$')
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            m = regex.match(dir)
            if m:
                g = m.groupdict()
                g['value'] = g['value'].replace('_', '-')
                print ' ... Scanning \"' + root + '/' + dir + '/\"'
                for f in os.listdir(os.path.join(root, dir)):
                    if f == (pkl + '.pkl'):
                        print ' * found pickle!'
                        df = pd.read_pickle(os.path.join(root, dir, f))
                        index.append(g['idx'] + ' ' + g['value'])
                        df_list.append(df)
    if df_list:
        df = pd.concat(df_list, keys=index, axis=0).reset_index(level=1)
        return df


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
def set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel):
    # set axis' labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # tight layout
    fig.set_tight_layout(True)
    # save df for post analysis
    df.to_pickle(out + 'df_' + desc + '.pkl')
    # save figure
    fig.savefig(out + 'fig_' + desc + '.pdf')

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
def plot_bar(df, desc, out, x, xlabel, y, ylabel):
    print '> Plotting ' + desc + ' (bar)'
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(kind='bar', ax=ax, rot=0)
    ax.legend_.remove()
    fig, ax = set_fig_and_save(df, fig, ax, desc, out, xlabel, ylabel)

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_box(df, desc, out, x, xlabel, y, ylabel):
    print '> Plotting ' + desc + ' (box)'
    fig, ax = plt.subplots(figsize=(8, 6))
    df.boxplot(ax=ax, grid=False)
    # ax.legend(['End-to-end delay (ms)'], loc=2)
    fig.set_tight_layout(True)
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
    gp = df.groupby(x)
    means = gp.mean()
    means.index = means.index.astype(int)
    errors = gp.std()
    fig, ax = plt.subplots(figsize=(8, 6))
    means.plot(kind='line', lw=2, fmt='b--s', ax=ax,
               yerr=errors, capsize=4, capthick=2)
    # ax.set_ylim(0, None)
    # xticks = [0, 1, 2, 3, 4, 5, 6]
    # ax.xaxis.set_ticks(xticks)
    ax.legend([ylabel], loc=2)
    fig.set_tight_layout(True)

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
    fig, ax = plot_bar(df, 'traffic_ratio', out, df.ratio, 'Traffic type',
                       df.index, 'Percentage of total traffic (\%)')

    return fig, ax


# ----------------------------------------------------------------------------#
def plot_reroute_scenario(df, out):
    print '> Analysis: Potting SDN reroute scenario'
    # plot mean delay of each app
    fig = plot_box(df, 'rr_scen', out, 'app', 'Application \#',
                   'lat', 'Mean Delay (ms)')

    return fig


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
