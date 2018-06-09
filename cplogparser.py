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
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

sim_dir = 'NOT_SET'
sim_type = 'NOT_SET'
sim_desc = 'NOT_SET'


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

    # dictionary of the various data df formatters
    read_function_map = {
        # atomic
        "atomic-energy":  format_atomic_energy_data,
        "atomic-op":  format_atomic_op_data,
        # usdn
        "pow":  format_usdn_pow_data,
        "app":  format_usdn_app_data,
        "sdn":  format_usdn_sdn_data,
        "node":  format_usdn_node_data,
        "icmp":  format_usdn_icmp_data,
        "join":  format_usdn_join_data,
    }

    try:
        # check the simulation sim_dir exists, and there is a log there
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
def plot_data(desc, type, dir, df_dict, plots):
    """Plot data according to required plot types."""
    global sim_desc, sim_type, sim_dir

    # required function for each plot type
    atomic_function_map = {
        # atomic
        'atomic_energy_v_hops': atomic_energy_v_hops,
        'atomic_op': atomic_op,
        # usdn
        'usdn_energy_v_hops': usdn_energy_v_hops,
        'usdn_prr_v_hops': usdn_prr_v_hops,
        'usdn_latency_v_hops': usdn_latency_v_hops,
        'usdn_join_time': usdn_join_time,
        'usdn_traffic_ratio': usdn_traffic_ratio,
        # atomic vs usdn
        'atomic_vs_usdn_join': atomic_vs_usdn_join,
        'atomic_vs_usdn_react': atomic_vs_usdn_react,
        'atomic_vs_usdn_configure': atomic_vs_usdn_configure,
        'atomic_vs_usdn_collect': atomic_vs_usdn_collect,
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        # atomic
        'atomic_energy_v_hops': ['atomic-energy'],
        'atomic_op': ['atomic-energy'],
        # usdn
        'usdn_energy_v_hops': ['pow', 'app'],
        'usdn_prr_v_hops': ['app'],
        'usdn_latency_v_hops': ['app'],
        'usdn_join_time': ['join'],
        'usdn_traffic_ratio': ['app', 'icmp'],
        # atomic vs usdn
        'atomic_vs_usdn_join': {'atomic': ['atomic-op'],
                                      'usdn': ['join', 'node']},
        'atomic_vs_usdn_react': {'atomic': ['atomic-op'],
                                       'usdn': ['sdn', 'node']},
        'atomic_vs_usdn_configure': {'atomic': ['atomic-op'],
                                     'usdn': ['sdn', 'node']},
        'atomic_vs_usdn_collect': {'atomic': ['atomic-op'],
                                         'usdn': ['sdn', 'node']}
    }

    sim_desc = desc
    sim_type = type
    sim_dir = dir

    print('> Do plots [' + ' '.join(plots) + '] for simulation: ' + desc)
    for plot in plots:
        try:
            if plot in atomic_function_map.keys():
                dfs = {}
                df_list = atomic_dict_map[plot][sim_type]
                dfs = {k: df_dict[k] for k in df_list if k in df_dict.keys()}
                if all(k in dfs for k in df_list):
                    atomic_function_map[plot](dfs)
                else:
                    raise Exception('ERROR: Required DFs not in dictionary!')
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
def format_usdn_pow_data(df):
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
def format_usdn_app_data(df):
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
def format_usdn_sdn_data(df):
    """Format sdn data."""
    print('> Read sdn sdn log')

    # Rearrange columns
    df = df.copy()
    df = df[['src', 'dest', 'typ', 'seq', 'time', 'status', 'id']]
    # Pivot table. Lose the 'mac' and 'id' column.
    df = df.pivot_table(index=['src', 'dest', 'typ', 'seq'],
                        columns=['status'],
                        aggfunc={'time': np.sum},
                        values=['time'])
    # TODO: not very elegant but it does the job
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df.columns = ['src', 'dest', 'typ', 'seq',
                  'in_t', 'out_t']
    # convert floats to ints
    df['dest'] = df['dest'].astype(int)
    df['seq'] = df['seq'].astype(int)
    # add a 'dropped' column
    df['drpd'] = df['in_t'].apply(lambda x: True if np.isnan(x) else False)
    # calculate the latency/delay and add as a column
    df['lat'] = (df['in_t'] - df['out_t'])/1000  # ms
    return df


# ----------------------------------------------------------------------------#
def format_usdn_node_data(df):
    """Format node data."""
    print('> Read sdn node log')
    # get the most common hops and degree for each node
    df = df.groupby('id')[['hops', 'degree']].agg(lambda x: x.mode())
    return df


# ----------------------------------------------------------------------------#
def format_usdn_icmp_data(df):
    """Format icmp data."""
    print('> Read sdn icmp log')
    # rearrage cols
    df = df[['level', 'module', 'type', 'code', 'id', 'time']]
    return df


# ----------------------------------------------------------------------------#
def format_usdn_join_data(df):
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
def atomic_op(df_dict):
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

    cpplot.plot_bar(df, 'atomic_op', sim_dir, x, y,
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
    cpplot.plot_bar(df, 'atomic_energy_v_hops', sim_dir,
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
    cpplot.plot_bar(df, 'usdn_energy_v_hops', sim_dir, x, y,
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
    cpplot.plot_bar(df, 'usdn_prr_v_hops', sim_dir, x, y,
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
    cpplot.plot_box(df, 'usdn_latency_v_hops', sim_dir, x, y,
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
    cpplot.plot_hist(df, 'usdn_join_time', sim_dir, x, y,
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
    cpplot.traffic_ratio(app_df, sdn_df, icmp_df, 'traffic_ratio', sim_dir)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_join(df_dict):
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
        df = df_dict['atomic-op'].copy()
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
    cpplot.plot_hist(df, 'atomic_vs_usdn_join', sim_dir, x, y,
                     xlabel=xlabel,
                     ylabel='Propotion of Nodes Joined',
                     color=color)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_react(df_dict):
    """Plot atomic vs usdn react times."""
    # parse the df
    if 'usdn' in sim_type:
        # get dfs
        df_node = df_dict['node'].copy().reset_index()
        df_node = df_node[df_node['hops'] > 0]
        df_sdn = df_dict['sdn'].copy()
        df_sdn = df_sdn[((df_sdn['typ'] == 'FTQ') | (df_sdn['typ'] == 'FTS'))
                        & (df_sdn['drpd'] == 0)]
        # separate ftq/fts
        ftq_df = df_sdn.loc[(df_sdn['typ'] == 'FTQ')]
        fts_df = df_sdn.loc[(df_sdn['typ'] == 'FTS')]
        ftq_df = ftq_df.drop('in_t', 1).rename(columns={'out_t': 'time'})
        fts_df = fts_df.drop('out_t', 1).rename(columns={'in_t': 'time'})
        # join them together and add a node id column based n FTQ src
        df = pd.concat([ftq_df, fts_df])
        df['id'] = np.where(df['typ'] == 'FTQ', df['src'], df['dest'])
        df = df.sort_values(['id', 'seq']).drop(['src', 'dest'], 1)
        df = df.merge(df_node, left_on='id', right_on='id')  # merge hops col
        df = df.pivot_table(index=['id', 'seq'],
                            columns=['typ'],
                            values=['time', 'hops'])
        # df is now multilevel, drop unanswered FTQs, and calc lat
        df = df[np.isfinite(df['hops']['FTS'])]
        df = df.drop(('hops', 'FTS'), 1)
        df['react_time'] = (df['time']['FTS'] - df['time']['FTQ'])/1000
        df.columns = df.columns.droplevel(0)
        df.columns = ['hops', 'FTQ', 'FTS', 'react_time']
        df['hops'] = df['hops'].astype(int)
    elif 'atomic' in sim_type:
        df = df_dict['atomic-op'].copy()
        df = df[df['op_type'] == 'RACT']
        print(df)
        df['react_time'] = df['c_time'].astype(int)
        df = df[df['react_time'] != 0]
        df = df[df['hops'] != 0]
    else:
        raise Exception('ERROR: Unknown sim type!')

    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='react_time')

    x = list(df.columns.values)  # x ticks are the column headers
    #  BOX
    # y = np.column_stack(df.transpose().values.tolist())  # need a list
    # cpplot.plot_box(df, 'atomic_vs_usdn_react', sim_dir, x, y,
    #                 xlabel='Hops', ylabel='End-to-end delay (ms)')
    #  LINE
    y = df.mean()
    cpplot.plot_line(df, 'atomic_vs_usdn_react', sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_configure(df_dict):
    """Plot atomic vs usdn react times."""
    # parse the df
    if 'usdn' in sim_type:
        # get dfs
        df_node = df_dict['node'].copy().reset_index()
        df_sdn = df_dict['sdn'].copy()
        # take only NSU and drop any dropped packets
        df = df_sdn[(df_sdn['typ'] == 'FTS') & (df_sdn['drpd'] == 0)]
        df = df.rename(columns={'lat': 'conf_time'})
        df = df.merge(df_node, left_on='dest', right_on='id')  # merge hops col
        print(df)
        df = df[df['hops'] != 0]
    elif 'atomic' in sim_type:
        df = df_dict['atomic-op'].copy()
        df = df[df['op_type'] == 'CONF']
        df['conf_time'] = df['c_time'].astype(int)
        df = df[df['conf_time'] != 0]
        df = df[df['hops'] != 0]
    else:
        raise Exception('ERROR: Unknown sim type!')

    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='conf_time')

    x = list(df.columns.values)  # x ticks are the column headers
    #  BOX
    # y = np.column_stack(df.transpose().values.tolist())  # need a list
    # cpplot.plot_box(df, 'atomic_vs_usdn_react', sim_dir, x, y,
    #                 xlabel='Hops', ylabel='End-to-end delay (ms)')
    #  LINE
    y = df.mean()
    cpplot.plot_line(df, 'atomic_vs_usdn_configure', sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_collect(df_dict):
    """Plot atomic vs usdn collect times."""
    if 'usdn' in sim_type:
        # copy dfs
        df_node = df_dict['node'].copy().reset_index()
        df_sdn = df_dict['sdn'].copy()
        # take only NSU and drop any dropped packets
        df = df_sdn[(df_sdn['typ'] == 'NSU') & (df_sdn['drpd'] == 0)]
        df = df.rename(columns={'lat': 'collect_time'})
        df = df[df['src'] != 1]
        df = df.merge(df_node, left_on='src', right_on='id')  # merge hops col
        df = df[df['hops'] != 0]
    elif 'atomic' in sim_type:
        df = df_dict['atomic-op'].copy()
        df = df[df['op_type'] == 'CLCT']
        df['collect_time'] = df['c_time'].astype(int)
        df = df[df['collect_time'] != 0]
        df = df[df['hops'] != 0]
    else:
        raise Exception('ERROR: Unknown types!')

    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='collect_time')
    x = list(df.columns.values)  # x ticks are the column headers
    # y = np.column_stack(df.transpose().values.tolist())  # need a list
    # cpplot.plot_box(df, 'atomic_vs_usdn_collect', sim_dir, x, y,
    #                 xlabel='Hops', ylabel='End-to-end delay (ms)')
    y = df.mean()
    cpplot.plot_line(df, 'atomic_vs_usdn_collect', sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
