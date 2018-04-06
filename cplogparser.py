#!/usr/bin/python
"""Parse logs and generate results.

This module parses cooja logs according to a list of required data.
"""
from __future__ import division
import os  # for makedir
import re  # regex
import sys
import matplotlib.pyplot as plt  # general plotting
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
from scipy.stats.mstats import mode

import cpplotter as cpplot

# from pprint import pprint

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
def scrape_data(datatype, log, dir, fmt, regex):
    """Scrape the main log for data."""
    # Print some information about what's being parsed
    info = 'Data: {0} | Log: {1} ' \
           '| Log Format: {2}'.format(datatype, log, fmt)
    print '-' * len(info)
    print info
    print '-' * len(info)

    # dictionary of the various data df formatters
    read_function_map = {
        "pow":  format_pow_data,
        "app":  format_app_data,
        "sdn":  format_sdn_data,
        "node": format_node_data,
        "icmp": format_icmp_data
    }

    try:
        # check the simulation directory exists, and there is a log there
        open(log, 'rb')
        # do the parsing
        print '*** Parsing log using ' + datatype + ' regex....'
        data_re = re.compile(regex)
        data_log = parse_log(log, dir + "log_" + datatype + ".log", data_re)
        if (os.path.getsize(data_log.name) != 0):
            data_df = csv_to_df(data_log)
            if datatype in read_function_map.keys():
                # do some formatting on the df
                data_df = read_function_map[datatype](data_df)
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
def analyze_data(df_dict):
    """Take the dfs generated from the main log and analyze."""
    print '*** Do some additional processing on the dataframes...'
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

    print '*** Node data summary...'
    print df_dict['node']


# ----------------------------------------------------------------------------#
def pickle_data(dir, data):
    """Save data by pickling it."""
    print '*** Pickling DataFrames ...'
    print data.keys()
    for k, v in data.items():
        print '> Saving ' + k
        if v is not None:
            v.to_pickle(dir + k + '_df.pkl')


# ----------------------------------------------------------------------------#
def plot_data(sim, dir, data, plots):
    """Plot data according to required plot types."""
    node_df = data['node']
    app_df = data['app']
    icmp_df = data['icmp']
    join_df = data['join']
    if 'sdn' in data.keys():
        sdn_df = data['sdn']
    else:
        sdn_df = None
    plot(sim, plots, dir, node_df=node_df, app_df=app_df, sdn_df=sdn_df,
         icmp_df=icmp_df, join_df=join_df)


# ----------------------------------------------------------------------------#
# Read logs
# ----------------------------------------------------------------------------#
def format_app_data(df):
    """Format application data."""
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
def format_icmp_data(df):
    """Format icmp data."""
    print '> Read icmp log'
    # TODO: Possibly do some processing?
    # print (df['type'] == 155).sum()
    # print (df['type'] == 200).sum()
    return df


# ----------------------------------------------------------------------------#
def format_sdn_data(sdn_df):
    """Format sdn data."""
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
                  'hops', 'in_t', 'out_t']
    # Set dest typ to int
    df.dest = df.dest.astype(int)
    # add a 'dropped' column
    df['drpd'] = df['in_t'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['in_t'] - df['out_t'])/1000  # FIXME: /1000 = ns -> ms

    return df


# ----------------------------------------------------------------------------#
def format_pow_data(df):
    """Format power data."""
    print '> Read power log'
    # get last row of each 'id' group and use the all_radio value
    df = df.groupby('id').last()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


# ----------------------------------------------------------------------------#
def format_node_data(df):
    """Format node data."""
    print '> Format node data'
    df = df.groupby('id')['rank', 'degree'].agg(lambda x: min(x.mode()))
    return df


# ----------------------------------------------------------------------------#
# Parse main log using regex
# ----------------------------------------------------------------------------#
def parse_log(file_from, file_to, pattern):
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
def csv_to_df(file):
    """Create df from csv."""
    df = pd.read_csv(file.name)
    # drop any ampty columns
    df = df.dropna(axis=1, how='all')
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
# General Plotting
# ----------------------------------------------------------------------------#
def plot(sim, plot_list, dir, node_df=None, app_df=None, sdn_df=None,
         icmp_df=None, join_df=None):
    """Process the data for all plottypes."""
    print '*** Do plots for simulation: ' + sim
    for plot in plot_list:
        # General plots
        # hops vs rdc
        if plot == 'hops_rdc':
            df = node_df.groupby('hops')['rdc'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            cpplot.plot_bar(df, plot, dir, df.index, df.rdc,
                            xlabel='Hops', ylabel='Radio duty cycle (\%)')
        # hops vs prr
        elif plot == 'hops_prr':
            df = node_df.groupby('hops')['prr'] \
                        .apply(lambda x: x.mean()) \
                        .reset_index() \
                        .set_index('hops')
            cpplot.plot_bar(df, plot, dir, df.index, df.prr,
                            xlabel='Hops', ylabel='PDR (\%)')
        # hops mean latency
        elif plot == 'hops_lat_mean':
            df = app_df[['hops', 'lat']].reset_index(drop=True)
            gp = df.groupby('hops')
            means = gp.mean()
            cpplot.plot_line(df, plot, dir, means.index, means.lat,
                             xlabel='Hops',
                             ylabel='Mean delay (ms)', steps=1.0)
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
            cpplot.plot_box(plot, dir, xticks, data,
                            xlabel='Hops', ylabel='End-to-end delay (ms)')
        # hops end to end latency
        elif plot == 'avg_lat':
            df = node_df['mean_lat']
            df = df.dropna(how='all')
            data = df.tolist()
            xticks = [sim]
            xlabel = 'Flowtable Lifetime (s)'
            # xlabel = 'Controller Update Period (s)'
            ylabel = 'End-to-end delay (ms)'
            cpplot.plot_box(plot, dir, xticks, data,
                            xlabel=xlabel, ylabel=ylabel)

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
            cpplot.plot_box(plot, dir, xticks, data,
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
            cpplot.plot_hist('c_join', dir,
                             df['controller'].tolist(), df.index.tolist(),
                             xlabel='Time (s)',
                             ylabel='Propotion of Nodes Joined')
            # plot_hist('dao_join', dir,
            #           df['dao'].tolist(), df.index.tolist(),
            #           xlabel='Time (s)', ylabel='Propotion of Nodes Joined')
            cpplot.plot_hist('dag_join', dir,
                             df['dag'].tolist(), df.index.tolist(),
                             xlabel='Time (s)',
                             ylabel='Propotion of Nodes Joined')
        # traffic ratio
        elif plot == 'net_tr':
            cpplot.traffic_ratio(app_df, sdn_df, icmp_df, dir)
