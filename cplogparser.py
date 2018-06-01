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
import traceback

import cpplotter as cpplot

from pprint import pprint

# Pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

directory = 'NOT_SET'
description = 'NOT_SET'


# ----------------------------------------------------------------------------#
# Helper functions
# ----------------------------------------------------------------------------#
def prr(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100


# ----------------------------------------------------------------------------#
# Main functions
# ----------------------------------------------------------------------------#
def scrape_data(datatype, log, dir, fmt, regex):
    """Scrape the main log for data."""
    # print(some information about what's being parsed
    info = 'Data: {0} | Log: {1} ' \
           '| Log Format: {2}'.format(datatype, log, fmt)
    print('-' * len(info))
    print(info)
    print('-' * len(info))

    # dictionary of the various data df formatters
    read_function_map = {
        # atomic
        "atomic-energy":  format_atomic_energy_data,
        "atomic-op":  format_atomic_op_data,
        # usdn
        "pow":  format_sdn_pow_data,
        "app":  format_sdn_app_data,
        "sdn":  format_sdn_sdn_data,
        "node":  format_sdn_node_data,
        "icmp":  format_sdn_icmp_data,
        "join":  format_sdn_join_data,
    }

    try:
        # check the simulation directory exists, and there is a log there
        open(log, 'rb')
        # do the parsing
        print('> Parsing log using ' + datatype + ' regex....')
        data_re = re.compile(regex)
        data_log = parse_log(log, dir + "log_" + datatype + ".log", data_re)
        if (os.path.getsize(data_log.name) != 0):
            data_df = csv_to_df(data_log)
            if datatype in read_function_map.keys():
                # do some formatting on the df
                data_df = read_function_map[datatype](data_df)
            else:
                data_df = None

            if data_df is not None:
                return data_df
            else:
                raise Exception('ERROR: Dataframe was None!')
        else:
            print('WARN: Log was empty')
    except Exception as e:
            traceback.print_exc()
            print(e)
            sys.exit(0)


# ----------------------------------------------------------------------------#
def pickle_data(dir, data):
    """Save data by pickling it."""
    print('> Pickling DataFrames ...')
    for k, v in data.items():
        print('> Saving ' + k)
        if v is not None:
            v.to_pickle(dir + k + '_df.pkl')


# ----------------------------------------------------------------------------#
def plot_data(sim, dir, df_dict, plots):
    """Plot data according to required plot types."""
    global description, directory

    # required function for each plot type
    atomic_function_map = {
        # atomic
        'atomic_energy_v_hops': atomic_energy_v_hops,
        'atomic_op_times': atomic_op_times,
        # usdn
        'usdn_energy_v_hops': usdn_energy_v_hops,
        'usdn_prr_v_hops': usdn_prr_v_hops,
        'usdn_latency_v_hops': usdn_latency_v_hops,
        'usdn_join_time': usdn_join_time,
        'usdn_traffic_ratio': usdn_traffic_ratio,
        # atomic vs usdn
        'atomic_vs_usdn_join_times': atomic_vs_usdn_join_times,
        'atomic_vs_usdn_react_times': atomic_vs_usdn_react_times,
        'atomic_vs_usdn_collect_times': atomic_vs_usdn_collect_times,
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        # atomic
        'atomic_energy_v_hops': ['atomic-energy'],
        'atomic_op_times': ['atomic-energy'],
        # usdn
        'usdn_energy_v_hops': ['pow', 'app'],
        'usdn_prr_v_hops': ['app'],
        'usdn_latency_v_hops': ['app'],
        'usdn_join_time': ['join'],
        'usdn_traffic_ratio': ['app', 'icmp'],
        # atomic vs usdn
        'atomic_vs_usdn_join_times': ['atomic-op', 'join'],
        'atomic_vs_usdn_react_times': ['atomic-op', 'sdn'],
        'atomic_vs_usdn_collect_times': ['atomic-op', 'sdn'],
    }

    # set plot descriptions
    description = sim
    directory = dir

    print('> Do plots [' + ' '.join(plots) + '] for simulation: ' + sim)
    for plot in plots:
        try:
            if plot in atomic_function_map.keys():
                dicts = {}
                dicts = {k: df_dict[k] for k in atomic_dict_map[plot]
                         if k in df_dict.keys()}
                atomic_function_map[plot](dicts)
            else:
                raise Exception('ERROR: No plot function!')
        except Exception as e:
                traceback.print_exc()
                print(e)
                sys.exit(0)


# ----------------------------------------------------------------------------#
# Read logs
# ----------------------------------------------------------------------------#
def format_atomic_energy_data(df):
    """Format atomic data."""
    print('> Read atomic log')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # rearrage other cols (and drop level/time)
    df = df[['id', 'module', 'op_type', 'n_phases', 'hops',
             'gon', 'ron', 'con', 'all_rdc', 'rdc']]
    # dump anything that isn't an PW log
    df = df[df['module'] == 'PW']

    return df


# ----------------------------------------------------------------------------#
def format_atomic_op_data(df):
    """Format atomic data."""
    print('> Read atomic log')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # rearrage other cols (and drop level/time)
    df = df[['id', 'module', 'op_type', 'hops', 'c_phase', 'n_phases',
             'c_time', 'op_duration']]
    # dump anything that isn't an OP log
    df = df[df['module'] == 'OP']
    # convert phase cols to ints
    df['c_phase'] = df['c_phase'].astype(int)
    df['n_phases'] = df['n_phases'].astype(int)

    return df


# ----------------------------------------------------------------------------#
def format_sdn_pow_data(df):
    """Format power data."""
    print('> Read sdn power log')
    # get last row of each 'id' group and use the all_radio value
    df = df.groupby('id').last()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')
    # rearrage cols
    df = df[['seqid', 'time',
             'all_radio', 'radio', 'all_tx', 'tx', 'all_listen', 'listen']]

    return df


# ----------------------------------------------------------------------------#
def format_sdn_app_data(df):
    """Format application data."""
    print('> Read sdn app log')
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
                                values='time')
    df = df.reset_index().rename(columns={'TX': 'txtime', 'RX': 'rxtime'})
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
def format_sdn_sdn_data(df):
    """Format sdn data."""
    print('> Read sdn sdn log')

    # Rearrange columns
    sdn_df = df[['src', 'dest', 'typ', 'seq', 'time',
                 'status', 'id', 'hops']]
    # Fixes settingwithcopywarning
    df = sdn_df.copy()
    # Fill in empty hop values for tx packets
    df['hops'] = sdn_df.groupby(['src', 'dest']).ffill().bfill()['hops']
    # Pivot table. Lose the 'mac' and 'id' column.
    df = df.pivot_table(index=['src', 'dest', 'typ', 'seq', 'hops'],
                        columns=['status'],
                        aggfunc={'time': np.sum},
                        values=['time'])
    # Get rid of the multiindex and rename the columns
    # TODO: not very elegant but it does the job
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df.columns = ['src', 'dest', 'typ', 'seq',
                  'hops', 'in_t', 'out_t']
    # convert floats to ints
    df['hops'] = df['hops'].astype(int)
    df['dest'] = df['dest'].astype(int)
    df['seq'] = df['seq'].astype(int)
    # add a 'dropped' column
    df['drpd'] = df['in_t'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['in_t'] - df['out_t'])/1000  # ms
    return df


# ----------------------------------------------------------------------------#
def format_sdn_node_data(df):
    """Format node data."""
    print('> Read sdn node log')
    # get the most common rank and degree for each node
    df = df.groupby('id')['rank', 'degree'].agg(lambda x: min(x.mode()))
    return df


# ----------------------------------------------------------------------------#
def format_sdn_icmp_data(df):
    """Format icmp data."""
    print('> Read sdn icmp log')
    # rearrage cols
    df = df[['level', 'module', 'type', 'code', 'id', 'time']]
    return df


# ----------------------------------------------------------------------------#
def format_sdn_join_data(df):
    """Format node data."""
    print('> Read sdn join log')
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
                # m = pattern.match(l)
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
# Atomic plotting
# ----------------------------------------------------------------------------#
def atomic_op_times(df_dict):
    """Plot atomic op times."""
    try:
        if 'atomic-energy' in df_dict:
            df = df_dict['atomic-energy']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)
    g = df.groupby('op_type')
    data = pd.DataFrame()

    for k, v in g:
        data[k] = pd.Series(v['con'].mean())

    # rearrage cols
    data = data[['NONE', 'CLCT', 'CONF', 'RACT', 'ASSC']]
    # rename cols
    data = data.rename(columns={'NONE': 'IND',
                                'CLCT': 'COLLECT',
                                'CONF': 'CONFIGURE',
                                'RACT': 'REACT',
                                'ASSC': 'ASSOCIATE'})
    x = list(data.columns.values)
    y = data.values.tolist()[0]

    cpplot.plot_bar(df, 'atomic_op_times', directory, x, y,
                    xlabel='Op Type', ylabel='Time(ms)')


# ----------------------------------------------------------------------------#
def atomic_energy_v_hops(df_dict):
    """Plot atomic energy vs hops."""
    try:
        if 'atomic-energy' in df_dict:
            df = df_dict['atomic-energy']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)
    g = df.groupby('hops')
    data = {}
    for k, v in g:
        # ignore the timesync (0 hops)
        if(k > 0):
            data[k] = v.groupby('id').last()['all_rdc'].mean()
    cpplot.plot_bar(df, 'atomic_energy_v_hops', directory,
                    data.keys(), data.values(),
                    xlabel='Hops', ylabel='Radio Duty Cycle (%)')


# ----------------------------------------------------------------------------#
# uSDN plotting
# ----------------------------------------------------------------------------#
def usdn_energy_v_hops(df_dict):
    """Plot usdn energy vs hops."""
    try:
        if 'app' in df_dict and 'pow' in df_dict:
            pow_df = df_dict['pow']
            app_df = df_dict['app']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)

    # Add hops to node_df. N.B. cols with NaN are always converted to float
    hops = app_df[['src', 'hops']].groupby('src').agg(lambda x: mode(x)[0])
    pow_df = pow_df.join(hops['hops'].astype(int))

    df = pow_df.groupby('hops')['all_radio']    \
               .apply(lambda x: x.mean()) \
               .reset_index()             \
               .set_index('hops')
    x = df.index.tolist()
    y = df['all_radio'].tolist()
    cpplot.plot_bar(df, 'usdn_energy_v_hops', directory, x, y,
                    xlabel='Hops',
                    ylabel='Radio duty cycle (%)')


# ----------------------------------------------------------------------------#
def usdn_prr_v_hops(df_dict):
    """Plot usdn energy vs hops."""
    try:
        if 'app' in df_dict:
            app_df = df_dict['app']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)

    # Get hops for each node. N.B. cols with NaN are always converted to float
    df = app_df[['src', 'hops']].groupby('src').agg(lambda x: mode(x)[0])
    # Calculate PRR
    df['prr'] = app_df.groupby('src')['drpd'] \
                      .apply(lambda x: prr(len(x), x.sum()))

    df = df.groupby('hops')['prr']    \
           .apply(lambda x: x.mean()) \
           .reset_index()             \
           .set_index('hops')
    x = df.index.tolist()
    y = df['prr'].tolist()
    cpplot.plot_bar(df, 'usdn_prr_v_hops', directory, x, y,
                    xlabel='Hops', ylabel='PDR (%)')


# ----------------------------------------------------------------------------#
def usdn_latency_v_hops(df_dict):
    """Plot usdn end-to-end latency vs hops."""
    try:
        if 'app' in df_dict:
            app_df = df_dict['app']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)

    # pivot table to...
    df = app_df.pivot_table(index=app_df.groupby('hops').cumcount(),
                            columns=['hops'], values='lat')
    df = df.dropna(how='all')  # drop rows with all NaN
    x = list(df.columns.values)  # x ticks are the column headers
    y = np.column_stack(df.transpose().values.tolist())  # need a list
    cpplot.plot_box(df, 'usdn_latency_v_hops', directory, x, y,
                    xlabel='Hops', ylabel='End-to-end delay (ms)')


# ----------------------------------------------------------------------------#
def usdn_join_time(df_dict):
    """Plot usdn controller and rpl-dag join times."""
    try:
        if 'join' in df_dict:
            join_df = df_dict['join']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)
    df = join_df.copy()
    df['time'] = join_df['time']/1000/1000
    # merge 'node' col into 'id' col, where the value in id is 1
    # FIXME: Not generic
    if 'node' in df:
        df.loc[df['id'] == 1, 'id'] = df['node']
        df = df.drop('node', 1)
    # drop the node/module/level columns
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
    # FIXME: Not generic
    if 'controller' in df:
        x = df['controller'].tolist()
    else:
        x = df['dag'].tolist()
    y = df.index.tolist()
    cpplot.plot_hist(df, 'usdn_join_time', directory, x, y,
                     xlabel='Time (s)',
                     ylabel='Propotion of Nodes Joined')


# ----------------------------------------------------------------------------#
def usdn_traffic_ratio(df_dict):
    """Plot traffic ratios."""
    try:
        if 'app' in df_dict and 'icmp' in df_dict:
            app_df = df_dict['app']
            icmp_df = df_dict['icmp']
            if 'sdn' in df_dict:
                sdn_df = df_dict['sdn']
            else:
                sdn_df = None
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)
    cpplot.traffic_ratio(app_df, sdn_df, icmp_df, 'traffic_ratio', directory)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_join_times(df_dict):
    """Plot atomic vs usdn join times."""
    # check we have the correct dicts
    try:
        if 'join' in df_dict:
            df = df_dict['join'].copy()
            type = 'usdn'
        elif 'atomic-op' in df_dict:
            df = df_dict['atomic-op'].copy()
            type = 'atomic'
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
            traceback.print_exc()
            sys.exit(0)
    if type is 'usdn':
        # get rows where node has joined controller
        df = df[df['controller'] == 1]
        # drop unecessary cols
        df['node'] = df['node'].astype(int)
        df = df[['node', 'time']].set_index('node').sort_index()
        # # convert time to ms
        df['time'] = df['time']/1000/1000
        df['time'] = df['time'].astype(int)
        xlabel = 'Time (s)'
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']
    if type is 'atomic':
        df = df[df['op_type'] == 'ASSC']
        df = df[['id', 'c_time']]
        df = df.rename(columns={'c_time': 'time'})
        df['time'] = df['time'].astype(int)/1000
        df = df.set_index('id').sort_index()
        df = df.iloc[1:]
        xlabel = 'Time (s)'
        color = list(plt.rcParams['axes.prop_cycle'])[1]['color']

    # plot the join times vs hops
    x = df['time'].tolist()
    y = df.index.tolist()
    cpplot.plot_hist(df, 'atomic_vs_usdn_join_times', directory, x, y,
                     xlabel=xlabel,
                     ylabel='Propotion of Nodes Joined',
                     color=color)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_react_times(df_dict):
    """Plot atomic vs usdn react times."""
    try:
        # check we have the correct dicts
        if 'sdn' in df_dict:
            df = df_dict['sdn'].copy()
            type = 'usdn'
        elif 'atomic-op' in df_dict:
            df = df_dict['atomic-op'].copy()
            type = 'atomic'
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
        # parse the df
        if type is 'usdn':
            ftq_df = df.loc[(df['typ'] == 'FTQ') & (df['drpd'] == 0)]
            ftq_df = ftq_df.drop('in_t', 1).rename(columns={'out_t': 'time'})
            fts_df = df.loc[(df['typ'] == 'FTS') & (df['drpd'] == 0)]
            fts_df = fts_df.drop('out_t', 1).rename(columns={'in_t': 'time'})
            df = pd.concat([ftq_df, fts_df])
            df = df.loc[((df['typ'] == 'FTQ') | (df['typ'] == 'FTS'))
                        & (df['drpd'] == 0)]
            df['id'] = np.where(df['typ'] == 'FTQ', df['src'], df['dest'])
            df = df.sort_values(['id', 'typ']).drop(['src', 'dest'], 1)
            df = df.set_index('id')
            hops = df.groupby('id').apply(lambda x: x.iloc[0]['hops'])
            df = df.pivot_table(index=['id'],
                                columns=['typ'],
                                values=['time'])
            df['id'] = df.index
            df.columns = df.columns.droplevel(0)
            df = df.reset_index().set_index('id')
            df['react_time'] = (df['FTS'] - df['FTQ'])/1000
            df['hops'] = hops
            df = df.pivot_table(index=df.groupby('hops').cumcount(),
                                columns=['hops'],
                                values='react_time',
                                fill_value=0)
        elif type is 'atomic':
            df = df[df['op_type'] == 'RACT']
            df['react_time'] = df['c_time']
            df = df[df['hops'] != 0]
            df = df.pivot_table(index=df.groupby('hops').cumcount(),
                                columns=['hops'],
                                values='react_time',
                                fill_value=0)
        else:
            raise Exception('ERROR: Unknown types!')

        x = list(df.columns.values)  # x ticks are the column headers
        y = np.column_stack(df.transpose().values.tolist())  # need a list
        cpplot.plot_box(df, 'atomic_vs_usdn_react_times', directory, x, y,
                        xlabel='Hops', ylabel='End-to-end delay (ms)')

    except Exception:
            traceback.print_exc()
            sys.exit(0)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_collect_times(df_dict):
    """Plot atomic vs usdn react times."""
    try:
        # check we have the correct dicts
        if 'sdn' in df_dict:
            df = df_dict['sdn'].copy()
            type = 'usdn'
        elif 'atomic-op' in df_dict:
            df = df_dict['atomic-op'].copy()
            type = 'atomic'
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
        # parse the df
        if type is 'usdn':
            df = df[(df['typ'] == 'NSU') & (df['drpd'] == 0)]
            df = df.rename(columns={'lat': 'collect_time'})
        elif type is 'atomic':
            df = df[df['op_type'] == 'CLCT']
            df['collect_time'] = df['c_time']
            df = df[df['hops'] != 0]
        else:
            raise Exception('ERROR: Unknown types!')

        df = df.pivot_table(index=df.groupby('hops').cumcount(),
                            columns=['hops'],
                            values='collect_time')
        x = list(df.columns.values)  # x ticks are the column headers
        y = np.column_stack(df.transpose().values.tolist())  # need a list
        cpplot.plot_box(df, 'atomic_vs_usdn_collect_times', directory, x, y,
                        xlabel='Hops', ylabel='End-to-end delay (ms)')

    except Exception:
            traceback.print_exc()
            sys.exit(0)
