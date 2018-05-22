#!/usr/bin/python
"""Parse logs and generate results.

This module parses cooja logs according to a list of required data.
"""
from __future__ import division
import os  # for makedir
import re  # regex
import sys
# import numpy as np  # number crunching
# import seaborn as sns  # fancy plotting
import pandas as pd  # table manipulation
# from scipy.stats.mstats import mode

import cpplotter as cpplot

# from pprint(import pprint

# Pandas options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
        "atomic":  format_atomic_data,
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
    # dictionary of the various plotters
    atomic_function_map = {
        'atomic_energy_v_hops': atomic_energy_v_hops,
        'atomic_op_times': atomic_op_times,
    }

    # set plot descriptions
    description = sim
    directory = dir

    print('> Do plots [' + ' '.join(plots) + '] for simulation: ' + sim)
    for plot in plots:
        print('> Plot ' + plot + '...')
        try:
            if plot in atomic_function_map.keys():
                atomic_function_map[plot](df_dict['atomic'])
            else:
                raise Exception('ERROR: No plot function!')
        except Exception as e:
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
                # m = pattern.match(l.replace('.-', '.'))
                m = pattern.match(l)
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
# General plotting
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
