#!/usr/bin/python
"""EWSN Packet Loss."""
import os
import traceback
import sys
import re
import numpy as np
import pandas as pd
# from pprint import pprint

# parse arguements
import argparse
import pickle   # for saving data

import cpplotter as cpplot
import matplotlib.pyplot as plt  # general plotting

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
# Helper functions
# ----------------------------------------------------------------------------#
def ratio(sent, received):
    """Calculate the packet receive rate of a node."""
    return (received/sent) * 100

# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    directory = '/home/mike/Results/atomic_mp2p_150219'
    out = '/home/mike/Results/'
    tmp_file = '/home/mike/Results/TMP'

    re_hb = '(?:(?P<heartbeat>\d+))\s<3+$'
    re_prefix = '^\[*(?:(?P<timestamp>.*))[\|\]]\s*'
    re_txrx = 'D:(?:\s*[ep:]+(?P<epoch>\d+)|\s+(?P<type>[FWDTRX-]+)\s+(?P<packet>.{1,16})|\s*[id:]+(?P<id>\d+)|\s*[s:]+(?P<src>\d+))*'

    # Cmd line args
    ap = argparse.ArgumentParser(prog='AtomicPy', description='Atomic Log Parser')
    ap.add_argument('--s', required=False, default=0, help='Save parsed logs')
    ap.add_argument('--l', required=False, default=0, help='Load saved logs')
    args = ap.parse_args()

    if args.l:
        print('.......... Load Results')
        results = pickle.load(open(out + 'results.pkl', 'rb'))
    else:
        # Regex
        regex = re_prefix + re_txrx
        print('.......... Parse Data')
        lp = LOG_PARSER()
        df = lp.parse_logs(directory, None, tmp_file, regex)

        print('.......... Format Data')
        # Get end-to-end delay for each packet
        df.set_index('timestamp', inplace=True, drop=False)
        df.sort_index(inplace=True)
        # df = df[df['id'] > 2]  # remove the first few packets (which are dropped because of the queue)
        # df_filtered = df.loc[((df['type'] == 'TX') | (df['type'] == 'RX'))]
        delay = df.pivot_table(index=['packet'],
                                    # columns=['type'],
                                    values=['timestamp'],
                                    aggfunc=lambda x: (x.max() - x.min())/np.timedelta64(1, 'ms'))
        # delay = delay.xs('timestamp', axis=1, drop_level=True)

        # Get the state for each node (cols) for each packet (index)
        table = df.pivot_table(index=['packet'],
                               columns=['node'],
                               values=['type'],
                               aggfunc=lambda x: ' '.join(x)).fillna('MISS')
        table = table.xs('type', axis=1, drop_level=True)

        print('.......... Generate Results')
        results = pd.DataFrame()
        results['packet'] = table.index
        results = results.set_index('packet')
        results['TX'] = table.apply(lambda row: row.to_string().count('TX'), axis=1)
        results['RX'] = table.apply(lambda row: row.to_string().count('RX'), axis=1)
        results['RTX'] = table.apply(lambda row: row.to_string().count('RTR'), axis=1)
        results['FWD'] = table.apply(lambda row: row.to_string().count('FWD'), axis=1)
        # results['NORX'] = df.apply(lambda row: row.to_string().count('MISS'), axis=1)
        results['epoch'] = df.groupby('packet')['epoch'].agg(lambda x: x.value_counts().index[0])  # returns the most common epoch
        results['src'] = df.groupby('packet')['src'].agg(lambda x: x.value_counts().index[0])  # returns a list of sources (they should be the same)
        results['id'] = df.groupby('packet')['id'].agg(lambda x: x.value_counts().index[0])  # returns a list of sources (they should be the same)
        results['lat'] = delay.groupby('packet')['timestamp'].agg(lambda x: x.value_counts().index[0])
        results = results.reset_index()
        results['status'] = results.apply(lambda row: packet_status(row), axis=1)
        if args.s:
            print('.......... Save Results')
            pickle.dump(results, open(out + 'results' + '.pkl', 'wb+'))

    print(results)
    # print('# Nodes (' + len(df['node'].unique()) + '): ' + str(df['node'].unique()))
    sent = (results['TX'] == 1).sum()
    received = (results['RX'] == 1).sum()
    total = results.shape[0]
    print('Total: ' + str(total))
    print('Sent: ' + str(sent))
    print('Received: ' + str(received))
    print('Retransmissions: ' + str((results['RTX'] == 1).sum()))
    print('Missed: ' + str(((results['RX'] == 0) & (results['TX'] == 1)).sum()))
    print(results.loc[((results['RX'] == 0) & (results['TX'] == 1))])
    print('Superfluous: ' + str(((results['TX'] == 0) & (results['RX'] == 1)).sum()))
    print(results.loc[((results['TX'] == 0) & (results['RX'] == 1))])

    print('.......... Graph Results')
    # Latency
    x = results.RTX
    y = y = results.lat.mean()
    cpplot.plot_line(results, 'tb_latency', out, x, y,
                     xlabel='Retransmissions', ylabel='End-to-end delay (ms)')
                     # ls='None')
    print('  ... LAT mean: ' + str(np.mean(y)))

    x = [0]
    y = ratio(sent, received)
    # PDR
    cpplot.plot_bar(results, 'tb_pdr', out, x, y,
                    xlabel='Packet Drop Rate (%)', ylabel='End-to-end PDR (%)')
    print('  ... PDR mean: ' + str(np.mean(y)))
