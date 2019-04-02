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
pd.set_option('display.max_rows', 100)
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
def reject_outliers(data, m=4):
    """Remove data outliers."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
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
        "app":  format_usdn_app_data,
        "sdn":  format_usdn_sdn_data,
        "node":  format_usdn_node_data,
        "icmp":  format_usdn_icmp_data,
        "join":  format_usdn_join_data,
        # common
        "pow":  format_energy_data,
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
        'atomic_op_times':    {'atomic'  : ['atomic-op']},
        # usdn
        'usdn_traffic_ratio': {'usdn'    : ['app', 'icmp']},
        # atomic vs usdn
        'association_v_time': {'atomic'  : ['atomic-op'],
                               'usdn'    : ['join'],
                               'sdn-wise': ['all']},
        'latency_v_hops':     {'atomic'  : ['atomic-op'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'pdr_v_hops':         {'atomic'  : ['atomic-op', 'atomic-energy'],
                               'usdn'    : ['sdn', 'node'],
                               'sdn-wise': ['all']},
        'energy_v_hops':      {'atomic'  : ['atomic-energy'],
                               'usdn'    : ['sdn', 'node', 'pow'],
                               'sdn-wise': ['all', 'pow']},
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
def format_atomic_energy_data(df):
    """Format atomic energy data."""
    print('> Format atomic energy data')
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
    print('> Format atomic op data')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # rearrage other cols (and drop level/time)
    df = df[['node', 'module', 'type', 'hops', 'c_phase', 'n_phases', 'lat', 'op_duration', 'active']]
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
def format_usdn_app_data(df):
    """Format usdn application data."""
    print('> Format usdn application data')
    # sort the table by src/dest/seq so txrx pairs will be next to each other
    # this fixes NaN hop counts being filled incorrectly
    df = df.sort_values(['src', 'dest', 'app', 'seq']).reset_index(drop=True)
    # Rearrange columns
    df = df[['node', 'state', 'src', 'dest', 'app', 'seq', 'time',
             'hops', 'type', 'module', 'level']]
    # fill in hops where there is a TX/RX
    df['hops'] = df.groupby(['src', 'dest', 'app', 'seq'])['hops'].apply(
                            lambda x: x.fillna(x.mean()))
    # pivot the table so we combine tx and rx rows for the same (src/dest/seq)
    df = df.bfill().pivot_table(index=['src', 'dest', 'app', 'seq', 'hops'],
                                columns=['state'],
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
    """Format usdn SDN data."""
    print('> Format usdn SDN data')

    # Rearrange columns
    df = df.copy()
    df = df[['src', 'dest', 'type', 'seq', 'time', 'state', 'hops']]
    df.state = df.state.str.replace('OUT', 'TX')
    df.state = df.state.str.replace('IN', 'RX')
    # Get head 'OUT' and tail 'IN' for 'CFG'
    index = ['src', 'dest', 'seq']
    mask = (df['type'] == 'CFG') & (df['state'] == 'TX')
    df[mask] = df[mask].groupby(index).head(1)
    mask = (df['type'] == 'CFG') & (df['state'] == 'RX')
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
                        columns=['state'],
                        aggfunc={'time': np.sum},
                        values=['time'])
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    # Add a 'dropped' column
    df['drpd'] = df['RX'].apply(lambda x: True if np.isnan(x) else False)
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
    global node_data
    print('> Read sdn node log')
    # get the most common hops and degree for each node
    df = df.groupby('node')[['hops', 'degree']].agg(lambda x: x.mode())
    df = df.reset_index()
    # node_df = df
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
    print('  ...Total: ' + str(df.shape[0]))
    print('  ...Sent: ' + str((df['TX'] > 0).sum()))
    print('  ...Received: ' + str((df['RX'] > 0).sum()))
    print('  ...Retransmissions: ' + str(df.received.str.count('rtx').sum()))
    print('  ...Missed: ' + str(df.received.str.count('missed').sum()))
    print(df[df['received'] == 'missed'])
    print('  ...Superfluous: ' + str(df.received.str.count('superfluous').sum()))
    print(df[df['received'] == 'superfluous'])


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

    return ret


# ----------------------------------------------------------------------------#
def create_packet_name(row):
    if (row.type == 'OPEN_PATH'):
        return str(row.type) + '_' + str(row.dest) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'REQUEST'):
        return str(row.type) + '_' + str(row.src) + '_' + str(row.origin) + '_' + str(row.seq)
    else:
        return str(row.type) + '_' + str(row.src) + '_' + str(row.dest) + '_' + str(row.seq)


# ----------------------------------------------------------------------------#
def create_packet_id(row):
    if (row.type == 'OPEN_PATH'):
        return str(row.dest) + '_' + str(row.origin) + '_' + str(row.seq)
    if (row.type == 'REQUEST'):
        return str(row.src) + '_' + str(row.origin) + '_' + str(row.seq)
    else:
        return str(row.src) + '_' + str(row.seq)


# ----------------------------------------------------------------------------#
def format_all_data(df):
    """Format all data."""
    print('> Format data')
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

    df.set_index('packet', inplace=True)
    df = get_packet_info(df)

    print_results(df)

    return df


# ----------------------------------------------------------------------------#
def format_energy_data(df):
    """Format energy data."""
    print('> Format energy')
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

    # rearrage cols
    data = data[['CONF', 'CLCT', 'RACT']]
    # # rename cols
    data = data.rename(columns={'CLCT': 'COLLECT',
                                'CONF': 'CONFIGURE',
                                'RACT': 'REACT'})
    x = list(data.columns.values)
    y = data.values.tolist()[0]

    print('  ... Op time mean: ' + str(np.mean(y)))
    print('  ... Op time median: ' + str(np.median(y)))
    print('  ... Op time max: ' + str(np.max(y)))

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
        color = list(plt.rcParams['axes.prop_cycle'])[2]['color']
        # df = df.set_index('node').sort_index()
        # df = df.iloc[1:]
    elif 'all' in df_name:
        # convert time to ms
        df = df.rename(columns={'maxtime': 'time', 'src': 'id'})
        df['time'] = df['time'].total_seconds()
        # set color
        color = list(plt.rcParams['axes.prop_cycle'])[1]['color']
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
                     ylabel='Proportion of Nodes',
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

    print('> Do latency_v_hops for ' + str(packets) + ' in \'df_' + df_name + '\'')

    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()

    # Filter df for packet types in packets
    df = df[df['type'].isin(packets)]

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
            df['lat'] = df.groupby('id').apply(lambda x: (x.maxtime.max() - x.mintime.min()).total_seconds())
            print(df.sort_values(by=['hops', 'mintime']))
        else:
            df = df.groupby(index, as_index=False).agg(aggfunc)

    # Reject outliers
    # df = df[(df['lat'] < 2000)]
    # df.lat = reject_outliers(df.lat)
    # df.dropna(subset=['lat'], inplace=True)

    # Find lat for each hop count
    df = df.pivot_table(index=df.groupby('hops').cumcount(),
                        columns=['hops'],
                        values='lat')
    x = list(df.columns.values)  # x ticks are the column headers
    y = df.mean()
    cpplot.plot_line(df, filename, sim_dir, x, y,
                     xlabel='Hops', ylabel='End-to-end delay (ms)')
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
        raise Exception('ERROR: No df!')
    if packets is None:
        raise Exception('ERROR: No packets to search for!')

    df = df_dict[df_name].copy()

    # Filter df for packet types in packets
    if 'type' in df:
        df = df[df['type'].isin(packets)]

    if 'drpd' not in df:
        df['drpd'] = np.where(df['received'] == 'missed', True, False)

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
    print('  ... PDR median: ' + str(np.median(y)))
    print('  ... PDR mode: ' + str(stats.mode(y)))


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
    df.set_index('node', inplace=True)

    if 'all' in df_dict:
        df_all = df_dict['all'][df_dict['all']['node'] != 1]
        # Filter df for packet types in packets
        if 'type' in df_all:
            df_all = df_all[df_all['type'].isin(packets)]
        df['hops'] = df_all.groupby('node').apply(lambda x: x.hops.value_counts().index[0])
    elif 'node' in df_dict:
        df['hops'] = df_dict['node']['hops']
    else:
        print('ERROR: No df to get hops from!')

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
