#!/usr/bin/python
"""EWSN Packet Loss."""
import os
import traceback
import sys
import re
import pandas as pd
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime
import difflib

# Pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.options.mode.chained_assignment = None


# ----------------------------------------------------------------------------#
def parse_log(file_from, file_to, pattern):
    """Parse a log using regex and save in new log."""
    # Let's us know this is the first line and we need to write a header.
    write_header = 1
    # open the files
    with open(file_from, 'r') as f:
        with open(file_to, 'w') as t:
            for l in f:
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
    df = pd.read_csv(file.name, parse_dates=True)
    df = df.dropna(axis=1, how='all')  # drop any empty columns
    return df


# ----------------------------------------------------------------------------#
def format_data(df):
    """Format neighbor data."""
    # remove whitespace
    # if 'packet' in df:
    #     df.packet = df.packet.str.replace(' ', '')
    df = df.dropna()
    if 'timestamp' in df:
        df.timestamp = pd.to_datetime(df.timestamp)
    if 'epoch' in df:
        df.epoch = df.epoch.astype(int)
    if 'src' in df:
        df.src = df.src.astype(int)
    if 'id' in df:
        df.id = df.id.astype(int)
    # if 'packet' in df:
        # df.packet = df[['packet', 'id']].apply(lambda x: '_'.join(x), axis=1)
        # df.packet = df.packet + "_" + df.id.map(str)
    return df


# ----------------------------------------------------------------------------#
def walklevel(some_dir, level=0):
    """Walk to a predefined level."""
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


# ----------------------------------------------------------------------------#
class LOG_PARSER:
    """LOG_PARSER class."""

    def parse_logs(self, dir, log, tmp, regex):
        """Parse logs according to the regex."""
        df_list = []
        data_re = re.compile(regex)
        try:
            # get table of log names to node_id
            id_df = pd.read_csv('/home/mike/Results/toshiba_tb_ids')
            id_df.set_index("name", inplace=True)
            # walk through directory structure
            for root, dirs, files in os.walk(dir):
                # print('  ... Files \"' + str(files) + '/\"')
                for file in files:
                    if(log is not None):
                        file = log
                    else:
                        file = dir + '/' + file
                    print('  ... Scanning \"' + file + '/\"')
                    fi = open(file, 'rb')
                    datafi = fi.read()
                    fi.close()
                    fo = open(file, 'wb')
                    fo.write(datafi.replace('\x00'.encode(), ''.encode()))
                    fo.close()
                    # check the dir exists, and there is a log there
                    open(file, 'rb')
                    # do the parsing
                    print('> Parsing log: ' + file)
                    data_log = parse_log(file, tmp, data_re)
                    # node_id = file.replace(dir + '/log_', '').strip('.txt')
                    node_id = id_df.loc[file.replace(dir + '/', '').strip('.txt')].id
                    if (os.path.getsize(data_log.name) != 0):
                        data_df = csv_to_df(data_log)  # convert from csv to df
                        data_df = format_data(data_df)  # format the df
                        if data_df is not None:
                            data_df['node'] = node_id
                            if(log is None):
                                df_list.append(data_df)
                            else:
                                return data_df
                        else:
                            raise Exception('ERROR: Dataframe was None!')
                    else:
                        print('WARN: No matching regex')
            if df_list:
                all_df = pd.concat(df_list, sort=True)
            else:
                raise Exception('ERROR: df_list was empty!')

            all_df = all_df.astype({"node": int})
            all_df = all_df.reset_index(drop=True)
            return all_df
        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e)
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(0)

# ----------------------------------------------------------------------------#
    def __init__(self):
        """Parse logs."""


# ----------------------------------------------------------------------------#
def select_rows(df, search_strings):
    """Return rows containing strings."""
    unq, IDs = np.unique(df, return_inverse=True)
    unqIDs = np.searchsorted(unq, search_strings)
    return df[((IDs.reshape(df.shape) == unqIDs[:, None, None])
               .any(-1)).all(0)]


# ----------------------------------------------------------------------------#
def packet_status(row):
    """Set the status of a packet based on the TX/RX"""
    ret = ''
    if row.TX == 1 and row.RX == 0:
        ret += 'missed'
    elif row.TX == 0 and row.RX == 1:
        ret += 'superflous'
    else:
        ret += 'correct'

    if row.RTX > 0:
        ret += ' - RTX'

    return ret


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    directory = '/home/mike/Results/atomic_mp2p_110219'
    tmp_file = '/home/mike/Results/TMP'

    re_hb = '(?:(?P<heartbeat>\d+))\s<3+$'
    re_prefix = '^\[*(?:(?P<timestamp>.*))[\|\]]\s*'
    re_txrx = 'D:(?:\s*[ep:]+(?P<epoch>\d+)|\s+(?P<type>[FWDTRX-]+)\s+(?P<packet>.{1,6})|\s*[id:]+(?P<id>\d+)|\s*[s:]+(?P<src>\d+))*'

    regex = re_prefix + re_txrx

    lp = LOG_PARSER()
    df = lp.parse_logs(directory, None, tmp_file, regex)
    # Get end-to-end delay for each packet
    df.set_index('timestamp', inplace=True, drop=False)
    df.sort_index(inplace=True)
    df_txrx = df.loc[((df['type'] == 'TX') | (df['type'] == 'RX'))]
    group = df_txrx.groupby('packet')
    for k, v in group:
        print(k, v)
    print(df_txrx[df_txrx['packet'] == '000a0a'])
    print('==================================')
    delay = df_txrx.bfill().pivot_table(index=['packet'],
                                # columns=['type'],
                                values=['timestamp'],
                                aggfunc=lambda x: (x.max() - x.min())/np.timedelta64(1, 'ms'))
    print(delay)
    # delay = delay.xs('timestamp', axis=1, drop_level=True)

    # Get the state for each node (cols) for each packet (index)
    table = df.pivot_table(index=['packet'],
                           columns=['node'],
                           values=['type'],
                           aggfunc=lambda x: ' '.join(x)).fillna('MISS')
    table = table.xs('type', axis=1, drop_level=True)

    results = pd.DataFrame()
    results['packet'] = table.index
    results = results.set_index('packet')

    print('==================================')
    results['TX'] = table.apply(lambda row: row.to_string().count('TX'), axis=1)
    results['RX'] = table.apply(lambda row: row.to_string().count('RX'), axis=1)
    results['RTX'] = table.apply(lambda row: row.to_string().count('RTR'), axis=1)
    results['FWD'] = table.apply(lambda row: row.to_string().count('FWD'), axis=1)
    results['NORX'] = table.apply(lambda row: row.to_string().count('MISS'), axis=1)
    results['epoch'] = df.groupby('packet')['epoch'].agg(lambda x: x.value_counts().index[0])  # returns the most common epoch
    results['src'] = df.groupby('packet')['src'].agg(lambda x: x.value_counts().index[0])  # returns a list of sources (they should be the same)
    results['lat'] = delay.groupby('packet')['timestamp'].agg(lambda x: x.value_counts().index[0])

    # Filter for bad results
    # for index, row in results.iterrows():
    #     if len(index) < 6:
    #         # try and find close matches
    #         same_epoch = results.loc[(results['epoch'] == row.epoch) & (results.index != index)]
    #         not_me = filter(lambda x: x is not index, same_epoch.index)
    #         matches = difflib.get_close_matches(index, not_me, n=1)
    #         print('----------------------------', end='')
    #         print(index + ' ' + str(row.epoch))
    #         print(same_epoch)
    #         print(matches)
    #         results.ix[matches[0]]['TX'] += row['TX']
    #         results.ix[matches[0]]['RX'] += row['RX']
    #         results.ix[matches[0]]['RTX'] += row['RTX']
    #         results.ix[matches[0]]['FWD'] += row['FWD']
    #         results.drop(index, inplace=True)
    #         # results.index = results.index.to_series().replace({index: matches[0]})
    #         # results = results.groupby(results.index, sort=False).sum()

    results = results.reset_index()
    results['status'] = results.apply(lambda row: packet_status(row), axis=1)

    print(results)

    print('Total: ' + str(results.shape[0]))
    print('Sent: ' + str((results['TX'] == 1).sum()))
    print('Received: ' + str((results['RX'] == 1).sum()))
    print('Retransmissions: ' + str((results['RTX'] == 1).sum()))
    print('No RX: ' + str(((results['RX'] == 0) & (results['TX'] == 1)).sum()))
    # print(results.loc[((results['RX'] == 0) & (results['TX'] == 1))])
    print('No TX: ' + str(((results['TX'] == 0) & (results['RX'] == 1)).sum()))
    print(results.loc[((results['TX'] == 0) & (results['RX'] == 1))])
