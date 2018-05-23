#!/usr/bin/python
"""Parse logs and generate results.

This module parses cooja logs according to a list of required data.
"""
from __future__ import division
import os  # for makedir
import re  # regex
import sys
import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
# from scipy.stats.mstats import mode
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
        "atomic":  format_atomic_data,
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
        'atomic_energy_v_hops': atomic_energy_v_hops,
        'atomic_op_times': atomic_op_times,
        'usdn_energy_v_hops': usdn_energy_v_hops,
    }

    # required dictionaries for each plotter
    atomic_dict_map = {
        'atomic_energy_v_hops': ['atomic'],
        'atomic_op_times': ['atomic'],
        'usdn_energy_v_hops': ['pow', 'node'],
    }

    # set plot descriptions
    description = sim
    directory = dir

    print('> Do plots [' + ' '.join(plots) + '] for simulation: ' + sim)
    for plot in plots:
        print('> Plot ' + plot + '...')
        try:
            if plot in atomic_function_map.keys():
                dicts = {}
                dicts = {k: df_dict[k] for k in atomic_dict_map[plot]}
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
def format_atomic_data(df):
    """Format atomic data."""
    print('> Read atomic log')
    # set epoch to be the index
    df.set_index('epoch', inplace=True)
    # rearrage other cols (and drop level/time)
    df = df[['id', 'module', 'op_type', 'n_phases', 'hops',
             'gon', 'ron', 'con', 'all_rdc', 'rdc']]

    return df


# ----------------------------------------------------------------------------#
def format_sdn_pow_data(df):
    """Format power data."""
    print('> Read sdn power log')
    # get last row of each 'id' group and use the all_radio value
    df = df.groupby('id').last()
    # need to convert all our columns to numeric values from strings
    df = df.apply(pd.to_numeric, errors='ignore')

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
    # rearrage cols
    df = df[['level', 'module', 'dag', 'id', 'time']]
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
def atomic_op_times(df):
    """Plot atomic op times."""
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
def atomic_energy_v_hops(df):
    """Plot atomic energy vs hops."""
    g = df.groupby('hops')
    data = {}
    for k, v in g:
        # ignore the timesync (0 hops)
        if(k > 0):
            data[k] = v.groupby('id').last()['all_rdc'].mean()
    cpplot.plot_bar(df, 'atomic_energy_v_hops', directory,
                    data.keys(), data.values(),
                    xlabel='Hops', ylabel='Radio Duty Cycle (\\%)')


# ----------------------------------------------------------------------------#
# uSDN plotting
# ----------------------------------------------------------------------------#
def usdn_energy_v_hops(df):
    """Plot usdn energy vs hops."""
    print(df)
    # df = df.groupby('hops')['rdc'] \
    #        .apply(lambda x: x.mean()) \
    #        .reset_index() \
    #        .set_index('hops')
    # x = df.index.tolist()
    # y = df['rdc'].tolist()
    # # print(x, y)
    # print('here2')
    # cpplot.plot_bar(df, 'usdn_energy_v_hops', directory, x, y,
    #                 xlabel='Hops',
    #                 ylabel='Radio duty cycle (\%)')
