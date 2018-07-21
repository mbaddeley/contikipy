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
pd.set_option('display.max_rows', 36)
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
        'atomic_op_times': atomic_op_times,
        # usdn
        'usdn_traffic_ratio': usdn_traffic_ratio,
        # atomic vs usdn
        'association_v_time': association_v_time,
        'latency_v_hops': latency_v_hops,
        'pdr_v_hops': pdr_v_hops,
        'energy_v_hops': energy_v_hops,
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        # atomic
        'atomic_op_times': {'atomic': ['atomic-op']},
        # usdn
        'usdn_traffic_ratio': {'usdn': ['app', 'icmp']},
        # atomic vs usdn
        'association_v_time': {'atomic': ['atomic-op'],
                               'usdn': ['join']},
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
    df = df[['node', 'module', 'type', 'n_phases', 'hops',
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
    df = df[['node', 'module', 'type', 'hops', 'c_phase', 'n_phases',
             'lat', 'op_duration', 'active']]
    # dump anything that isn't an OP log
    df = df[df['module'] == 'OP']
    # fill in a dropped col
    df['drpd'] = np.where((df['active'] == 1) & (df['lat'] == 0), True, False)
    # convert to ints
    df['c_phase'] = df['c_phase'].astype(int)
    df['n_phases'] = df['n_phases'].astype(int)
    df['active'] = df['active'].astype(int)
    # HACK: Convert CONF active nodes to 1;
    df.loc[df.type == 'CONF', ['active']] = 1
    return df


# ----------------------------------------------------------------------------#
def format_usdn_pow_data(df):
    """Format power data."""
    print('> Read sdn power log')
    # get last row of each 'node' group and use the all_radio value
    df = df.groupby('node').mean()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')
    # rearrage cols
    df = df[['all_rdc', 'rdc']]
    # make 'node' a col
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
    df = df[['node', 'status', 'src', 'dest', 'app', 'seq', 'time',
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
# def create_usdn_id(row):
#     if row['type'] == ():
#         val =
#     else:
#         val =
#     return val


# ----------------------------------------------------------------------------#
def format_usdn_sdn_data(df):
    """Format sdn data."""
    print('> Read sdn sdn log')

    # Rearrange columns
    df = df.copy()
    df = df[['src', 'dest', 'type', 'seq', 'time', 'status', 'hops']]
    # Get head 'OUT' and tail 'IN' for 'CFG'
    index = ['src', 'dest', 'seq']
    mask = (df['type'] == 'CFG') & (df['status'] == 'OUT')
    df[mask] = df[mask].groupby(index).head(1)
    mask = (df['type'] == 'CFG') & (df['status'] == 'IN')
    df[mask] = df[mask].groupby(index).tail(1)
    # Create an 'id' col based on the actively participating node
    # (src for uplink, dest for uplink)
    uplink = (df['type'] == 'FTQ') | (df['type'] == 'NSU') | \
             (df['type'] == 'DAO')
    df['id'] = np.where(uplink, df['src'], df['dest']).astype(int)
    df = df[df['type'].notnull()]
    # Fill in the hops
    index = ['src', 'dest', 'type', 'seq']
    df['hops'] = df.groupby(index, sort=False)['hops'].apply(
                            lambda x: x.ffill().bfill())
    index = ['src', 'dest']
    # Fill NaN hops with mode using 'id'
    df['hops'] = df.groupby('id')['hops'] \
                   .transform(lambda x: x.fillna(x.mean()))
    df = df.pivot_table(index=['src', 'dest', 'type', 'seq', 'hops', 'id'],
                        columns=['status'],
                        aggfunc={'time': np.sum},
                        values=['time'])
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    # Add a 'dropped' column
    df['drpd'] = df['IN'].apply(lambda x: True if np.isnan(x) else False)
    # Rename columns
    df.columns = ['src', 'dest', 'type', 'seq', 'hops', 'id',
                  'in_t', 'out_t', 'drpd']
    # calculate the latency/delay and add as a column
    df['lat'] = (df['in_t'] - df['out_t'])/1000  # ms
    # convert floats to ints
    df['src'] = df['src'].astype(int)
    df['dest'] = df['dest'].astype(int)
    df['seq'] = df['seq'].astype(int)
    df['hops'] = df['hops'].astype(int)

    return df


# ----------------------------------------------------------------------------#
def format_usdn_node_data(df):
    """Format node data."""
    print('> Read sdn node log')
    # get the most common hops and degree for each node
    df = df.groupby('node')[['hops', 'degree']].agg(lambda x: x.mode())
    df = df.reset_index()
    return df


# ----------------------------------------------------------------------------#
def format_usdn_icmp_data(df):
    """Format icmp data."""
    print('> Read sdn icmp log')
    # rearrage cols
    df = df[['level', 'module', 'type', 'code', 'node', 'time']]
    return df


# ----------------------------------------------------------------------------#
def format_usdn_join_data(df):
    """Format node data."""
    print('> Read sdn join log')
    # rearrage cols
    df = df[['level', 'module', 'dag', 'dao',
             'controller', 'node', 'id', 'time']]
    df['id'] = df['id'].fillna(df['node'])
    df = df[['dag', 'dao', 'controller', 'id', 'time']]
    df = (df.set_index(['id', 'time'])
            .stack()
            .reorder_levels([2, 0, 1])
            .reset_index(name='a')
            .drop('a', 1)
            .rename(columns={'level_0': 'type'}))
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
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'atomic_op_times'
    df = df_dict[df_name].copy()
    print(df[df['type'] == 'CONF'])
    # Filter df for packet types in packets
    if 'type' in df and packets is not None:
        df = df[df['type'].isin(packets)]
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

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Op Type', ylabel='Time(ms)')


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
def association_v_time(df_dict, **kwargs):
    """Plot atomic vs usdn join times."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'association_v_time'

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    # TODO: Make this more abstract
    if 'atomic' in df_name:
        # convert time to ms
        df = df.rename(columns={'lat': 'time', 'node': 'id'})
        df['time'] = df['time'].astype(int)/1000
        # set color
        color = list(plt.rcParams['axes.prop_cycle'])[1]['color']
        # df = df.set_index('node').sort_index()
        # df = df.iloc[1:]
    else:
        # convert time to ms
        df['time'] = df['time']/1000/1000
        df['time'] = df['time']
        # set color
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # plot the join times vs hops
    x = df['time'].tolist()
    y = df['id'].astype(int).tolist()
    cpplot.plot_hist(df, filename, sim_dir, x, y,
                     xlabel='Time (s)',
                     ylabel='Propotion of Nodes Joined',
                     color=color)
    print('  ... Association mean: ' + str(np.mean(x)))
    print('  ... Association median: ' + str(np.median(x)))
    print('  ... Association max: ' + str(np.max(x)))


# ----------------------------------------------------------------------------#
def latency_v_hops(df_dict, **kwargs):
    """Plot latency vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    index = kwargs['index'] if 'index' in kwargs else None
    aggfunc = kwargs['aggfunc'] if 'aggfunc' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'latency_v_hops'

    print('> Do latency_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    # For multiple packets we need aggregate based on a function
    # (e.g. {'lat': sum})
    if len(packets) > 1 and index is not None and aggfunc is not None:
        index.append('hops')
        df = df.groupby(index, as_index=False).aggregate(aggfunc)

    # HACK: Removes some ridiculous outliers at hop 4
    df = df[(df['lat'] < 2000)]

    # Find lat for each hop count
    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='lat')
    x = list(df.columns.values)  # x ticks are the column headers
    y = df.mean()
    # HACK because we aren't getting 5 hops
    # if 'atomic' in df_name:
    #     s = pd.Series([34.000], index=[5])
    #     x.append(5)
    #     y = y.append(s)
    #     print(df_name, y)
    cpplot.plot_line(df, filename, sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
    print('  ... LAT mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def pdr_v_hops(df_dict, **kwargs):
    """Plot pdr vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'pdr_v_hops'

    print('> Do pdr_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    df_pdr = df.groupby('hops')['drpd'].apply(lambda x: ratio(len(x), x.sum()))
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
    filename = kwargs['file'] if 'file' in kwargs else 'energy_v_hops'

    print('> Do energy_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    g = df.groupby('hops')
    data = {}
    for k, v in g:
        data[k] = v.groupby('node').last()['all_rdc'].mean()

    x = data.keys()
    y = data.values()

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Hops', ylabel='Radio Duty Cycle (%)')
    print('  ... RDC mean: ' + str(np.mean(y)))
