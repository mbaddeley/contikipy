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
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

sim_dir = 'NULL'
sim_type = 'NULL'
sim_desc = 'NULL'


# ----------------------------------------------------------------------------#
# Helper functions
# ----------------------------------------------------------------------------#
def ratio(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100


# ----------------------------------------------------------------------------#
# Main functions
# ----------------------------------------------------------------------------#
def scrape_data(datatype, log, dir, regex):
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
        print('> Parsing log: ' + log)
        print('> Match regex: ' + datatype)
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
        'atomic_op_times': atomic_op_times,
        # usdn
        'usdn_energy_v_hops': usdn_energy_v_hops,
        'usdn_pdr_v_hops': usdn_pdr_v_hops,
        'usdn_latency_v_hops': usdn_latency_v_hops,
        'usdn_join_time': usdn_join_time,
        'usdn_traffic_ratio': usdn_traffic_ratio,
        # atomic vs usdn
        'atomic_vs_usdn_join': atomic_vs_usdn_join,
        'latency_v_hops': latency_v_hops,
        'pdr_v_hops': pdr_v_hops,
        'energy_v_hops': energy_v_hops,
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        # atomic
        'atomic_energy_v_hops': {'atomic': ['atomic-energy']},
        'atomic_op_times': {'atomic': ['atomic-op']},
        # usdn
        'usdn_energy_v_hops': {'usdn': ['pow', 'app']},
        'usdn_pdr_v_hops': {'usdn': ['app']},
        'usdn_latency_v_hops': {'usdn': ['app']},
        'usdn_join_time': {'usdn': ['join']},
        'usdn_traffic_ratio': {'usdn': ['app', 'icmp']},
        # atomic vs usdn
        'atomic_vs_usdn_join': {'atomic': ['atomic-op'],
                                'usdn': ['join', 'node']},
        'latency_v_hops': {'atomic': ['atomic-op'],
                           'usdn': ['sdn', 'node']},
        'pdr_v_hops': {'atomic': ['atomic-op', 'atomic-energy'],
                       'usdn': ['sdn', 'node']},
        'energy_v_hops': {'atomic': ['atomic-energy'],
                          'usdn': ['sdn', 'node', 'pow']},
    }

    sim_desc = desc
    sim_type = type
    sim_dir = dir

    for plot in plots:
        try:
            if plot in atomic_function_map.keys():
                dfs = {}
                df_list = atomic_dict_map[plot][sim_type]
                dfs = {k: df_dict[k] for k in df_list if k in df_dict.keys()}
                if all(k in dfs for k in df_list):
                    if plots[plot] is None:
                        atomic_function_map[plot](dfs)
                    else:
                        atomic_function_map[plot](dfs, **plots[plot])
                else:
                    raise Exception('ERROR: Required DFs not in dictionary '
                                    + 'for plot [' + plot + ']')
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
    df = df[['id', 'module', 'type', 'n_phases', 'hops',
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
    df = df[['id', 'module', 'type', 'hops', 'c_phase', 'n_phases',
             'lat', 'op_duration', 'active']]
    # dump anything that isn't an OP log
    df = df[df['module'] == 'OP']
    # fill in a dropped col
    df['drpd'] = np.where((df['active'] == 1) & (df['lat'] == 0), True, False)
    # convert to ints
    df['c_phase'] = df['c_phase'].astype(int)
    df['n_phases'] = df['n_phases'].astype(int)
    df['active'] = df['active'].astype(int)
    return df


# ----------------------------------------------------------------------------#
def format_usdn_pow_data(df):
    """Format power data."""
    print('> Read sdn power log')
    # get last row of each 'id' group and use the all_radio value
    df = df.groupby('id').mean()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')
    # rearrage cols
    df = df[['all_rdc', 'rdc']]
    # make 'id' a col
    df = df.reset_index()

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
             'hops', 'type', 'module', 'level']]
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
    df = df[['src', 'dest', 'type', 'seq', 'time', 'status', 'id', 'hops']]
    # Get head 'OUT' and tail 'IN' for 'CFG'
    index = ['src', 'dest', 'seq']
    mask = (df['type'] == 'CFG') & (df['status'] == 'OUT')
    df[mask] = df[mask].groupby(index).head(1)
    mask = (df['type'] == 'CFG') & (df['status'] == 'IN')
    df[mask] = df[mask].groupby(index).tail(1)
    # print(df[(df['type'] == 'CFG') & (df['dest'] == 24)])
    # Fill in the hops
    index = ['src', 'dest', 'type', 'seq']
    df['hops'] = df.groupby(index, sort=False)['hops'].apply(
                            lambda x: x.ffill().bfill())
    df = df.pivot_table(index=['src', 'dest', 'type', 'seq', 'hops'],
                        columns=['status'],
                        aggfunc={'time': np.sum},
                        values=['time']).bfill()
    # TODO: not very elegant but it does the job
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    df.columns = ['src', 'dest', 'type', 'seq', 'hops', 'in_t', 'out_t']
    # convert floats to ints
    df['dest'] = df['dest'].astype(int)
    df['seq'] = df['seq'].astype(int)
    df['hops'] = df['hops'].astype(int)
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
    df = df.reset_index()
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
def atomic_op_times(df_dict, **kwargs):
    """Plot atomic op times."""
    df = df_dict['atomic-op'].copy()
    g = df.groupby('type')
    data = pd.DataFrame()

    for k, v in g:
        data[k] = pd.Series(v['op_duration'].mean())

    # # rearrage cols
    # data = data[['NONE', 'CLCT', 'CONF', 'RACT', 'ASSC']]
    # # rename cols
    # data = data.rename(columns={'NONE': 'IND',
    #                             'CLCT': 'COLLECT',
    #                             'CONF': 'CONFIGURE',
    #                             'RACT': 'REACT',
    #                             'ASSC': 'ASSOCIATE'})
    x = list(data.columns.values)
    y = data.values.tolist()[0]

    cpplot.plot_bar(df, 'atomic_op', sim_dir, x, y,
                    xlabel='Op Type', ylabel='Time(ms)')


# ----------------------------------------------------------------------------#
def atomic_energy_v_hops(df_dict, **kwargs):
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
def usdn_energy_v_hops(df_dict, **kwargs):
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

    df = pow_df.groupby('hops')['all_rdc']    \
               .apply(lambda x: x.mean()) \
               .reset_index()             \
               .set_index('hops')
    x = df.index.tolist()
    y = df['all_rdc'].tolist()
    cpplot.plot_bar(df, 'usdn_energy_v_hops', sim_dir, x, y,
                    xlabel='Hops',
                    ylabel='Radio duty cycle (%)')


# ----------------------------------------------------------------------------#
def usdn_pdr_v_hops(df_dict, **kwargs):
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
    df['pdr'] = app_df.groupby('src')['drpd'] \
                      .apply(lambda x: ratio(len(x), x.sum()))

    df = df.groupby('hops')['pdr']    \
           .apply(lambda x: x.mean()) \
           .reset_index()             \
           .set_index('hops')
    x = df.index.tolist()
    y = df['pdr'].tolist()
    cpplot.plot_bar(df, 'usdn_pdr_v_hops', sim_dir, x, y,
                    xlabel='Hops', ylabel='PDR (%)')


# ----------------------------------------------------------------------------#
def usdn_latency_v_hops(df_dict, **kwargs):
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
def usdn_join_time(df_dict, **kwargs):
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
def usdn_traffic_ratio(df_dict, **kwargs):
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

    cpplot.plot_bar(app_df, sdn_df, icmp_df, 'traffic_ratio', sim_dir)


# ----------------------------------------------------------------------------#
def atomic_vs_usdn_join(df_dict, **kwargs):
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
        df = df[df['type'] == 'ASSC']
        df = df[['id', 'lat']]
        df = df.rename(columns={'lat': 'time'})
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
# def latency_v_hops_ract(df_dict, **kwargs):
#     """Plot atomic vs usdn react times."""
#     # parse the df
#     if 'usdn' in sim_type:
#         # get dfs
#         df_node = df_dict['node'].copy().reset_index()
#         df_node = df_node[df_node['hops'] > 0]
#         df_sdn = df_dict['sdn'].copy()
#         df_sdn = df_sdn[((df_sdn['type'] == 'FTQ') | (df_sdn['type'] == 'FTS'))
#                         & (df_sdn['drpd'] == 0)]
#         # separate ftq/fts
#         ftq_df = df_sdn.loc[(df_sdn['type'] == 'FTQ')]
#         fts_df = df_sdn.loc[(df_sdn['type'] == 'FTS')]
#         ftq_df = ftq_df.drop('in_t', 1).rename(columns={'out_t': 'time'})
#         fts_df = fts_df.drop('out_t', 1).rename(columns={'in_t': 'time'})
#         # join them together and add a node id column based n FTQ src
#         df = pd.concat([ftq_df, fts_df])
#         df['id'] = np.where(df['type'] == 'FTQ', df['src'], df['dest'])
#         df = df.sort_values(['id', 'seq']).drop(['src', 'dest'], 1)
#         df = df.merge(df_node, left_on='id', right_on='id')  # merge hops col
#         df = df.pivot_table(index=['id', 'seq'],
#                             columns=['type'],
#                             values=['time', 'hops'])
#         # df is now multilevel, drop unanswered FTQs, and calc lat
#         df = df[np.isfinite(df['hops']['FTS'])]
#         df = df.drop(('hops', 'FTS'), 1)
#         df['react_time'] = (df['time']['FTS'] - df['time']['FTQ'])/1000
#         df.columns = df.columns.droplevel(0)
#         df.columns = ['hops', 'FTQ', 'FTS', 'react_time']
#         df['hops'] = df['hops'].astype(int)
#         df = df[(df['hops'] > 0) & (df['hops'] <= 5)]
#     elif 'atomic' in sim_type:
#         df = df_dict['atomic-op'].copy()
#         df = df[df['type'] == 'RACT']
#         df = df[df['lat'] != 0]
#         df['react_time'] = df['lat'].astype(int)
#         df = df[df['hops'] != 0]
#     else:
#         raise Exception('ERROR: Unknown sim type!')
#
#     df = df.pivot_table(index=df.groupby('hops').cumcount(),
#                         columns=['hops'],
#                         values='react_time')
#
#     # min = df.min().tolist()
#     # max = df.max().tolist()
#     x = list(df.columns.values)  # x ticks are the column headers
#     y = df.mean()  # df.mode().transpose()[0]
#     e = None
#     cpplot.plot_line(df, 'latency_v_hops_ract', sim_dir, x, y, errors=e,
#                      xlabel='Hops', ylabel='End-to-end delay (ms)')


# ----------------------------------------------------------------------------#
def latency_v_hops(df_dict, **kwargs):
    """Plot latency vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    ack = kwargs['packets'] if 'ack_packet' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'latency'

    print('> Do latency_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    df = df[df['lat'] != 0]
    df = df[df['hops'] != 0]

    # HACK: Removes some ridiculous outliers at hop 4
    df = df[(df['lat'] < 2000)]

    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='lat')
    x = list(df.columns.values)  # x ticks are the column headers

    y = df.mean()
    cpplot.plot_line(df, filename, sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
    print('  ... LAT mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def pdr_v_hops(df_dict, **kwargs):
    """Plot pdr vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    ack = kwargs['packets'] if 'ack_packet' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'latency'

    print('> Do pdr_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    df_pdr = df.groupby('hops')['drpd'] \
               .apply(lambda x: ratio(len(x), x.sum()))
    df_pdr = df_pdr.groupby('hops')           \
                   .apply(lambda x: x.mean()) \
                   .reset_index()             \
                   .set_index('hops')

    x = df_pdr.index.tolist()
    y = df_pdr['drpd'].tolist()
    cpplot.plot_bar(df_pdr, filename, sim_dir, x, y,
                    xlabel='Hops', ylabel='End-to-end PDR (%)')
    print('  ... PDR mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def energy_v_hops(df_dict, **kwargs):
    """Plot energy vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    ack = kwargs['packets'] if 'ack_packet' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'latency'

    print('> Do energy_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    g = df.groupby('hops')
    data = {}
    for k, v in g:
        data[k] = v.groupby('id').last()['all_rdc'].mean()

    x = data.keys()
    y = data.values()

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Hops', ylabel='Radio Duty Cycle (%)')
    print('  ... RDC mean: ' + str(np.mean(y)))
