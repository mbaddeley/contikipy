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

# from pprint(import pprint

# Matplotlib settings for graphs (need texlive-full, ghostscript and dvipng)
plt.rc('font', family='sans-serif', weight='bold')
plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\\usepackage{cmbright}')
plt.rc('text.latex', preamble='\\usepackage{sfmath}')
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
    # dictionary of the various plotters
    atomic_function_map = {
        'atomic_associate_time':  atomic_associate_time,
        'atomic_collect_time':  atomic_collect_time,
    }
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
def atomic_associate_time(df):
    """Plot atomic associate data."""
    df = df.loc[df['op_type'] == 'ASSC']
    print(df)

    return df


# ----------------------------------------------------------------------------#
def atomic_collect_time(df):
    """Plot atomic associate data."""
    df = df.loc[df['op_type'] == 'CLCT']
    print(df)

    return df



# ----------------------------------------------------------------------------#
# Additional processing
# ----------------------------------------------------------------------------#
def prr(sent, dropped):
    """Calculate the packet receive rate of a node."""
    return (1 - dropped/sent) * 100
