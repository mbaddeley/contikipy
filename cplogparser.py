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
from scipy import stats
import traceback

import cpplotter as cpplot

# from pprint import pprint


# Pandas options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

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
def reject_outliers(data, m=2):
    """Remove data outliers."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    print(mdev)
    s = d/mdev if mdev else 0
    return data[s < m]


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
        "sdn":  format_sdn_data,
        "icmp":  format_icmp_data,
        # common
        "node":  format_node_data,
        "pow":  format_energy_data,
        "join": format_join_data,
        "all":  format_all_data,
    }

    try:
        # check the simulation sim_dir exists, and there is a log there
        # open(log, 'rb')
        # do the parsing
        print('> Scraping log using regex: \'' + datatype + '\'')
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
            raise Exception('ERROR: Log was empty!')
    except Exception as e:
        traceback.print_exc()
        print(e)
        sys.exit(0)


# ----------------------------------------------------------------------------#
def plot_data(desc, type, dir, df_dict, plots):
    """Plot data according to required plot types."""
    global sim_desc, sim_type, sim_dir

    # required function for each plot type
    atomic_function_map = {
        # atomic
        'atomic_op_times'   : atomic_op_times,
        # common
        'association_v_time': association_v_time,
        'latency_v_hops'    : latency_v_hops,
        'pdr_v_hops'        : pdr_v_hops,
        'energy_v_hops'     : energy_v_hops,
        'latency'           : graph_latency,
        'pdr'               : graph_pdr,
        'traffic_ratio'     : graph_traffic_ratio
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        # atomic
        'atomic_op_times':    {'atomic'  : ['atomic-op']},
        # atomic vs usdn
        'association_v_time': {'atomic'  : ['atomic-op'],
                               'usdn'    : ['join'],
                               'sdn-wise': ['join']},
        'latency_v_hops':     {'atomic'  : ['atomic-op'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'pdr_v_hops':         {'atomic'  : ['atomic-op', 'atomic-energy'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'energy_v_hops':      {'atomic'  : ['atomic-energy'],
                               'usdn'    : ['sdn', 'node', 'pow'],
                               'sdn-wise': ['all', 'pow']},
        'latency':            {'atomic'  : ['atomic-op'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'pdr':                {'atomic'  : ['atomic-op'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'traffic_ratio':      {'usdn'    : ['sdn', 'icmp'],
                               'sdn-wise': ['all']},
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
                # print(dfs)
                if all(k in dfs for k in df_list):
                    if plots[plot] is None:
                        atomic_function_map[plot](dfs)
                    else:
                        atomic_function_map[plot](dfs, **plots[plot])
                else:
                    # print(dfs)
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
def create_packet_name(row):
    if (row.type == 'OPEN_PATH'):
        return str(row.type) + '_' + str(row.dest) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'REQUEST'):
        return str(row.type) + '_' + str(row.src) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'FTS'):
        return str(row.type) + '_' + str(row.dest) + '_' + str(row.seq)
    else:
        return str(row.type) + '_' + str(row.src) + '_' + str(row.dest) + '_' + str(row.seq)


# ----------------------------------------------------------------------------#
def create_packet_id(row):
    if (row.type == 'OPEN_PATH'):
        return str(row.dest) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'REQUEST'):
        return str(row.src) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'FTS'):
        return str(row.dest) + '_' + str(row.seq)
    else:
        return str(row.src) + '_' + str(row.seq)


# ----------------------------------------------------------------------------#
def get_packet_info(df):
    """Get the status and latency of each packet."""

    # Get the status of each packet
    table = df.pivot_table(index=['packet'],
                           columns=['node'],
                           values=['state'],
                           aggfunc=lambda x: ' '.join(x)).fillna('MISS')
    table = table.xs('state', axis=1, drop_level=True)
    ret = pd.DataFrame()
    ret['packet'] = table.index
    ret.set_index('packet', inplace=True)
    if df.state.str.match('TX').any():
        ret['TX'] = table.apply(lambda row: row.to_string().count('TX'), axis=1)
    if df.state.str.match('RX').any():
        ret['RX'] = table.apply(lambda row: row.to_string().count('RX'), axis=1)
    if df.state.str.match('RTX').any():
        ret['RTX'] = table.apply(lambda row: row.to_string().count('RTR'), axis=1)
    if df.state.str.match('FWD').any():
        ret['FWD'] = table.apply(lambda row: row.to_string().count('FWD'), axis=1)

    ret['received'] = ret.apply(lambda row: packet_status(row), axis=1)

    # Get back some of the columns we have lost
    ret['node'] = df.groupby('packet')['src'].agg(lambda x: x.value_counts().index[0])
    ret['type'] = df.groupby('packet')['type'].agg(lambda x: x.value_counts().index[0])
    ret['id'] = df.groupby('packet')['id'].agg(lambda x: x.value_counts().index[0])
    ret['src'] = df.groupby('packet')['src'].agg(lambda x: x.value_counts().index[0])
    ret['dest'] = df.groupby('packet')['dest'].agg(lambda x: x.value_counts().index[0])
    ret['seq'] = df.groupby('packet')['seq'].agg(lambda x: x.value_counts().index[0])  # returns a list of sources (they should be the same)
    ret['origin'] = df.groupby('packet')['origin'].agg(lambda x: x.value_counts().index[0])
    ret['target'] = df.groupby('packet')['target'].agg(lambda x: x.value_counts().index[0])
    ret['hops'] = df.groupby('packet').apply(lambda x: x['hops'].max())

    # Get the latency for each packet
    table = df.pivot_table(index=['packet'],
                           values=['time'],
                           aggfunc=lambda x: (x.max() - x.min())/np.timedelta64(1, 'ms'))
    ret['lat'] = table.groupby('packet')['time'].agg(lambda x: x.value_counts().index[0])
    ret['mintime'] = df.groupby(['packet']).apply(lambda x: x['time'].min())  # min TX time
    ret['maxtime'] = df.groupby(['packet']).apply(lambda x: x['time'].max())  # max RX time
    ret.loc[~ret['received'].str.contains('correct'), 'lat'] = 0.0

    # print(ret[ret['received'] != 'correct'])

    return ret


# ----------------------------------------------------------------------------#
def packet_status(row):
    """Set the received status of a packet based on the TX/RX"""
    ret = ''
    if 'TX' in row and 'RX' in row:
        if row.RX == 0:
            ret = 'missed'
        elif row.TX > 1:
            ret = 'correct'
        elif row.RX > 1:
            ret = 'superflous'
        else:
            ret += 'correct'
    else:
        print('ERROR: Could not determine packet status')

    return ret


# ----------------------------------------------------------------------------#
def print_results(df):
    """Print the final results dataframe."""
    missed = df.received.str.count('missed').sum()
    superfluous = df.received.str.count('superfluous').sum()
    print('  ... Total: ' + str(df.shape[0]))
    print('  ... Sent: ' + str((df['TX'] > 0).sum()))
    print('  ... Received: ' + str((df['RX'] > 0).sum()))
    print('  ... Retransmissions: ' + str(df.received.str.count('rtx').sum()))
    print('  ... Missed: ' + str(missed))
    if(missed):
        print(df[df['received'] == 'missed'])
    print('  ... Superfluous: ' + str(superfluous))
    # if(superfluous):
    #     print(df[df['received'] == 'superfluous'])

    # TODO can do this with a pivot_table and sum aggfunc
    unique_packets = df.type.unique()
    for p in unique_packets:
        print('  ... No. ' + p + ': ' + str(df.type.str.count(p).sum()))


# ----------------------------------------------------------------------------#
def format_atomic_energy_data(df):
    """Format atomic energy data."""
    print('  > Format atomic energy data')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # rearrage other cols (and drop level/time)
    df = df[['node', 'module', 'type', 'n_phases', 'hops', 'gon', 'ron', 'con', 'all_rdc', 'rdc']]
    # dump anything that isn't an PW log

    return df


# ----------------------------------------------------------------------------#
def format_atomic_op_data(df):
    """Format atomic data."""
    print('  > Format atomic op data')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # dump anything that isn't an OP log
    df = df[df['module'] == 'OP']
    # rearrage other cols (and drop level/time)
    df = df[['node', 'module', 'type', 'hops', 'c_phase', 'n_phases', 'lat', 'op_duration', 'active']]
    # fill in a dropped col
    df['drpd'] = np.where((df['active'] == 1) & (df['lat'] == 0), True, False)
    df['received'] = df['drpd'].apply(lambda x: 'missed' if x is True else 'correct')
    # convert to ints
    df['c_phase'] = df['c_phase'].astype(int)
    df['n_phases'] = df['n_phases'].astype(int)
    df['active'] = df['active'].astype(int)
    # HACK: Convert CONF active nodes to 1;
    df.loc[df.type == 'CONF', ['active']] = 1
    # HACK: Look into why lat times have increased
    # df['lat'] = df['lat']/3
    print('  ... Number of Nodes: ' + str(df.node.unique().shape[0]))
    print('  ... Max Hops: ' + str(df.hops.max()))
    print('  ... Mean Hops: ' + str(df.hops.mean()))
    return df


# ----------------------------------------------------------------------------#
def format_sdn_data(df):
    """Format usdn SDN data."""
    print('  > Format SDN data')

    # Rearrange columns
    df = df.copy()
    # print(df)
    df.state = df.state.str.replace('OUT', 'TX')
    df.state = df.state.str.replace('IN', 'RX')

    if 'time' in df:
        df.time = df.time * 1000  # convert to ns
        df.time = pd.to_datetime(df.time)
    if 'node' in df:
        df.node = df.node.astype(int)
    if 'src' in df:
        df.src = df.src.astype(int)
    if 'dest' in df:
        df.dest = df.dest.astype(int)
    if 'seq' in df:
        df.seq = df.seq.fillna(value=0)
        df.seq = df.seq.astype(int)
    if 'hops' in df:
        df.hops = df.hops.fillna(value=0)
        df.hops = df.hops.astype(int)
    if 'origin' not in df:
        # FIXME
        df['origin'] = df.src
    if 'target' not in df:
        # FIXME
        df['target'] = 1
    if 'packet' not in df:
        df['packet'] = df.apply(lambda row: create_packet_name(row), axis=1)
    if 'id' not in df:
        df['id'] = df.apply(lambda row: create_packet_id(row), axis=1)

    df.set_index('packet', inplace=True)
    df = get_packet_info(df)

    # HACK drop hops col so we can merge with node df
    # df.drop(columns={'hops'}, inplace=True)

    print_results(df)

    return df


# ----------------------------------------------------------------------------#
def format_icmp_data(df):
    """Format icmp data."""
    print('> Read icmp log')
    # rearrage cols
    df = df[['level', 'module', 'type', 'code', 'node', 'time']]
    return df


# ----------------------------------------------------------------------------#
def format_all_data(df):
    """Format all data."""
    print('  > Format data')
    df = df.copy()

    df.set_index('time', inplace=True, drop=False)
    df.sort_index(inplace=True)

    # If we have any filters on for hops then some fields may be NaN
    df.dropna(subset=['hops'], inplace=True)

    if 'time' in df:
        df.time = df.time * 1000  # convert to ns
        df.time = pd.to_datetime(df.time)
    if 'seq' in df:
        df.seq = df.seq.astype(int)
    if 'origin' in df:
        df.origin = df.origin.fillna(value=0)
        df.origin = df.origin.astype(int)
    if 'target' in df:
        df.target = df.target.fillna(value=0)
        df.target = df.target.astype(int)
    if 'packet' not in df:
        df['packet'] = df.apply(lambda row: create_packet_name(row), axis=1)
    if 'id' not in df:
        df['id'] = df.apply(lambda row: create_packet_id(row), axis=1)

    df = get_packet_info(df)

    print_results(df)

    # # HACK drop hops col so we can merge with node df
    # df.drop(columns={'hops'}, inplace=True)

    return df


# ----------------------------------------------------------------------------#
def format_energy_data(df):
    """Format energy data."""
    print('  > Format energy')
    # get last row of each 'node' group and use the all_radio value
    df = df.groupby('node').mean()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')
    # rearrage cols
    df = df[['all_rdc', 'rdc']]
    # make 'node' a col
    df = df.reset_index()

    print('  ...RDC: ' + str(df.all_rdc.mean()))

    return df


# ----------------------------------------------------------------------------#
def format_node_data(df):
    """Format join data."""
    global node_data
    print('  > Format node')
    # get the most common hops and degree for each node
    df = df.groupby('node')[['hops', 'degree']].agg(lambda x: x.mode())
    df = df.reset_index()
    print('  ... Number of Nodes: ' + str(df.shape[0]))
    print('  ... Max Hops: ' + str(df.hops.max()))
    print('  ... Mean Hops: ' + str(df.hops.mean()))
    print('  ... Max Degree: ' + str(df.degree.max()))
    print('  ... Mean Degree: ' + str(df.degree.mean()))

    return df


# ----------------------------------------------------------------------------#
def format_join_data(df):
    """Format node data."""
    print('  > Format join')
    # rearrage cols
    if 'dag' in df:
        df = df[['level', 'module', 'dag', 'dao',
                 'controller', 'node', 'id', 'time']]
        df['id'] = df['id'].fillna(df['node'])
        df = df[['dag', 'dao', 'controller', 'id', 'time']]
    else:
        df.drop(columns={'node', 'level', 'module'}, inplace=True)

    # Create a 'type' column, which shows what sort of association it is (rpl, controller, etc.)
    df = (df.set_index(['id', 'time'])
            .stack()
            .reorder_levels([2, 0, 1])
            .reset_index(name='a')
            .drop('a', 1)
            .rename(columns={'level_0': 'type'}))
    df['time'] = pd.to_timedelta(df.time * 1000).dt.total_seconds()  # convert from ns to ms
    return df


# ----------------------------------------------------------------------------#
# Parse main log using regex
# ----------------------------------------------------------------------------#
def parse_log(file_from, file_to, pattern):
    """Parse a log using regex and save in new log."""
    # Let's us know this is the first line and we need to write a header.
    write_header = 1
    # open the files
    with open(file_from, 'r') as f:
        with open(file_to, 'w') as t:
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
    df = pd.read_csv(file.name, parse_dates=True)
    df = df.dropna(axis=1, how='all')  # drop any empty columns
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
    # print(df[df['type'] == 'CLCT'])
    # Filter df for packet types in packets
    if 'type' in df and packets is not None:
        df = df[df['type'].isin(packets)]
    g = df.groupby('type')
    data = pd.DataFrame()
    for k, v in g:
        data[k] = pd.Series(v['op_duration'].mean())

    # # rename cols
    data = data.rename(columns={'CONF': 'CONFIGURE',
                                'CLCT': 'COLLECT',
                                'RACT': 'REACT'})
    data = data[['CONFIGURE', 'COLLECT', 'REACT']]
    x = list(data.columns.values)
    y = data.values.tolist()[0]

    print('  ... Op time mean: ' + str(np.mean(y)))
    print('  ... Op time median: ' + str(np.median(y)))
    print('  ... Op time max: ' + str(np.max(y)))

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Op Type', ylabel='Time(ms)')


# ----------------------------------------------------------------------------#
def graph_traffic_ratio(df_dict, **kwargs):
    """Graph ratio of traffic for packets."""
    try:
        if 'sdn' in df_dict and 'icmp' in df_dict:
            sdn_df = df_dict['sdn']
            icmp_df = df_dict['icmp']
        else:
            raise Exception('ERROR: Correct df(s) not in dict!')
    except Exception:
        traceback.print_exc()
        sys.exit(0)

    cpplot.plot_bar(sdn_df, icmp_df, 'traffic_ratio', sim_dir)


# ----------------------------------------------------------------------------#
def association_v_time(df_dict, **kwargs):
    """Plot atomic vs usdn join times."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'association_v_time'

    print('> Do association_v_time for ' + str(packets) + ' in \'df_' + df_name + '\'')

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    # TODO: Make this more abstract
    if 'atomic' in df_name:
        # convert time to ms
        df = df.rename(columns={'lat': 'time', 'node': 'id'})
        df['time'] = df['time'].astype(int)/1000
        df = df[['type', 'id', 'time']].sort_values(by='time').reset_index()

    # plot the join times vs hops
    x = df['time'].tolist()
    y = df['id'].astype(int).tolist()

    cpplot.plot_hist(df, filename, sim_dir, x, y,
                     xlabel='Time (s)',
                     ylabel='Proportion of Nodes')

    print('  ... Association mean: ' + str(np.mean(x)))
    print('  ... Association std_dev: ' + str(np.std(x)))
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

    print('> Do latency_v_hops for ' + str(packets) + ' in \'df_' + df_name + '\'')

    df = df_dict[df_name].copy()
    # Filter df for packet types in packets
    df = df[df['type'].isin(packets)]
    if(df.shape[0] == 0):
        print('Error: No packets in DF!')

    # Only correctly received packets
    if 'received' in df:
        df = df[(df['received'] == 'correct') | (df['received'] == 'rtx')]

    # For multiple packets we need aggregate based on a function (e.g. {'lat': sum})
    if len(packets) > 1 and index is not None and aggfunc is not None:
        index.append('hops')
        if 'between' in aggfunc:
            df = df[df.duplicated(subset=['id'], keep=False)]
            df['packet'] = df.index
            # We need to set the index to id so we can create a new column from the groupby
            df.set_index('id', inplace=True)
            df['lat'] = df.groupby('id').apply(lambda x: (x.maxtime.max() - x.mintime.min()).total_seconds()*1000)
        else:
            df = df.groupby(index, as_index=False).agg(aggfunc)

    # Find lat for each hop count
    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='lat')
    x = df.columns.values.astype(int)  # x ticks are the column headers
    y = df.mean()
    # HACK:
    if 5 not in x:
        x = np.append(x, 5)
        y = np.append(y, y.mean())
    cpplot.plot_line(df, filename, sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)', errors=True)
    print('  ... LAT mean: ' + str(np.mean(y)))
    print('  ... LAT median: ' + str(np.median(y)))
    print('  ... LAT mode: ' + str(stats.mode(y)))


# ----------------------------------------------------------------------------#
def pdr_v_hops(df_dict, **kwargs):
    """Plot pdr vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'pdr_v_hops'

    print('> Do pdr_v_hops for ' + str(packets) + ' in ' + df_name)

    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()

    print('  ... Total for df (' + df_name + ') = ' + str(df.shape[0]))

    if 'type' in df:
        df = df[df['type'].isin(packets)]

    missed = df.received.str.count('missed').sum()
    received = df.received.str.count('correct').sum()

    if 'drpd' not in df:
        df['drpd'] = np.where(df['received'] == 'missed', True, False)

    total = df.shape[0]

    print('  ... Total: ' + str(total))
    print('  ... Received: ' + str(received))
    print('  ... Missed: ' + str(missed))
    print('  ... Total PDR: ' + str(ratio(total, missed)))

    df_pdr = df.groupby('hops')['drpd'].apply(lambda x: ratio(len(x), x.sum()))
    df_pdr = df_pdr.groupby('hops')           \
                   .apply(lambda x: x.mean()) \
                   .reset_index()             \
                   .set_index('hops')

    x = df_pdr.index.tolist()
    y = df_pdr['drpd'].tolist()

    cpplot.plot_bar(df_pdr, filename, sim_dir, x, y,
                    xlabel='Hops', ylabel='End-to-end PDR (%)')


# ----------------------------------------------------------------------------#
def energy_v_hops(df_dict, **kwargs):
    """Plot energy vs hops."""
    df_name = kwargs['df'] if 'df' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'energy_v_hops'

    print('> Do energy_v_hops for ' + df_name)

    df = df_dict[df_name].copy()
    df.set_index('node', inplace=True)

    if 'all' in df_dict:
        df_all = df_dict['all'][df_dict['all']['node'] != 1]
        df['hops'] = df_all.groupby('node').apply(lambda x: x.hops.value_counts().index[0])
    elif 'node' in df_dict:
        df['hops'] = df_dict['node']['hops']

    # HACK: Why isnt the filter filtering for hops between 1 and 5?
    df = df[(df.hops <= 5) & (df.hops >= 1)]

    g = df.groupby('hops')
    data = {}
    for k, v in g:
        data[k] = v.groupby('node').last()['all_rdc'].mean()

    x = data.keys()
    y = data.values()

    if y is not list:
        y = list(y)

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Hops', ylabel='Radio Duty Cycle (%)')
    print('  ... RDC mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def graph_latency(df_dict, **kwargs):
    """Graph end-to-end delay."""

    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'latency'
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else packets

    print('> Do packet_latency for ' + str(packets) + ' in ' + df_name)

    df = df_dict[df_name].copy()
    if packets is not None:
        df = df[df['type'].isin(packets)]
        # x = packets
        # y = df.groupby('type')['lat'].mean()

    x = [str(xlabel)]
    y = [df[(df['received'] == 'correct') | (df['received'] == 'superfluous')].lat.mean()]
    # cpplot.plot_bar(df, filename, sim_dir, x, y,
    #                 xlabel='Packet Type', ylabel='End-to-end delay (ms)')
    cpplot.plot_line(df, filename, sim_dir, x, y,
                     xlabel='Number of Nodes', ylabel='End-to-end delay (ms)',
                     prefix=xlabel + '_',
                     errors=True)

    print('  ... LAT mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def graph_pdr(df_dict, **kwargs):
    global hlp
    """Graph end-to-end PDR."""

    df_name = kwargs['df'] if 'df' in kwargs else None
    packets = kwargs['packets'] if 'packets' in kwargs else None
    filename = kwargs['file'] if 'file' in kwargs else 'pdr'
    xlabel = kwargs['xlabel'] if 'xlabel' in kwargs else packets

    df = df_dict[df_name].copy()

    if packets is not None:
        print('> Do packet_pdr for ' + str(packets) + ' in ' + df_name)
        df = df[df['type'].isin(packets)]
    else:
        print('> Do packet_pdr for ALL_PACKETS in ' + df_name)

    total = df.shape[0]
    missed = df.received.str.count('missed').sum()
    received = df.received.str.count('correct').sum()
    print('  ... Total: ' + str(total))
    print('  ... Received: ' + str(received))
    print('  ... Missed: ' + str(missed))
    print('  ... Total PDR: ' + str(ratio(total, missed)))

    if packets is not None:
        if 'drpd' not in df:
            df['drpd'] = np.where(df['received'] == 'missed', True, False)
        df_pdr = df.groupby('type')['drpd'].apply(lambda x: ratio(len(x), x.sum()))
        df_pdr = df_pdr.groupby('type')           \
                       .apply(lambda x: x.mean()) \
                       .reset_index()             \
                       .set_index('type')
        x = df_pdr.index.tolist()
        y = df_pdr['drpd'].tolist()
    else:
        x = [str(xlabel)]
        y = [ratio(total, missed)]

    cpplot.plot_bar(df, filename, sim_dir, x, y,
                    xlabel='Packet Type', ylabel='End-to-end PDR (%)')
