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
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

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
    df = df.dropna()
    if 'timestamp' in df:
        print(df.timestamp)
        df.timestamp = pd.to_datetime(df.timestamp)
    if 'epoch' in df:
        df.epoch = df.epoch.astype(int)
    if 'src' in df:
        df.src = df.src.astype(int)
    if 'id' in df:
        df.id = df.id.astype(int)
    # if 'packet' in df:
        # df.packet = df[['id', 'src']].apply(lambda x: '_'.join(str(x)), axis=1)
        # df.packet = df['id'].astype(str) + '_' + df['src'].astype(str)
    print(df)
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
            i = 0
            # get table of log names to node_id
            id_df = pd.read_csv('/home/mike/Results/toshiba_tb_ids')
            id_df.set_index("name", inplace=True)
            # walk through directory structure
            for root, dirs, files in os.walk(dir):
                # print('  ... Files \"' + str(files) + '/\"')
                for file in files:
                    i = i + 1
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
            print('> Finished parsing ' + str(i) + ' files')
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
class HELPER:
    """HELPER class."""

    def reject_outliers(self, data, m=2):
        """Remove data outliers."""
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0
        return data[s < m]

    def ratio(self, sent, received):
        """Calculate the packet receive rate of a node."""
        return (received/sent) * 100

# ----------------------------------------------------------------------------#
    def __init__(self):
        """A helper function library."""


# ----------------------------------------------------------------------------#
# MAIN
# ----------------------------------------------------------------------------#
hlp = HELPER()


def generate_results(df):
    """
    Generate the results DataFrame from a DataFrame containing TX/RX info
    for each node with the packet id as the index, as well as the
    """
    # Get the latency for each packet
    delay = df.pivot_table(index=['packet'],
                           values=['timestamp'],
                           aggfunc=lambda x: (x.max() - x.min())/np.timedelta64(1, 'ms'))
    # Get the state for each node (cols) for each packet (index)
    table = df.pivot_table(index=['packet'],
                           columns=['node'],
                           values=['type'],
                           aggfunc=lambda x: ' '.join(x)).fillna('MISS')
    table = table.xs('type', axis=1, drop_level=True)
    # Generate the results df
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
    results['received'] = results.apply(lambda row: packet_status(row), axis=1)

    results['lat'] = delay.groupby('packet')['timestamp'].agg(lambda x: x.value_counts().index[0])
    results.loc[~results['received'].str.contains('correct'), 'lat'] = 0.0
    results = results.reset_index()
    print(results)
    return results


# ----------------------------------------------------------------------------#
def print_results(df):
    """Print the final results dataframe."""
    sent = (df['TX'] == 1).sum()
    received = (df['RX'] == 1).sum()
    total = df.shape[0]
    missed = ((df['RX'] == 0) & (df['TX'] == 1)).sum()
    superfluous = ((df['TX'] == 0) & (df['RX'] == 1)).sum()
    print('Total: ' + str(total))
    print('Sent: ' + str(sent))
    print('Received: ' + str(received))
    print('Retransmissions: ' + str((df['RTX'] == 1).sum()))
    print('Missed: ' + str(missed))
    if(missed):
        print(df.loc[((df['RX'] == 0) & (df['TX'] == 1))])
    print('Superfluous: ' + str(superfluous))
    if(superfluous):
        print(df.loc[((df['TX'] == 0) & (df['RX'] == 1))])


# ----------------------------------------------------------------------------#
def graph_latency(df, out):
    """Graph end-to-end delay."""
    x = df.RTX
    y = df[(df['TX'] == 1) & (df['RX'] == 1)].lat.mean()
    cpplot.plot_line(df, 'tb_latency', out + '/', x, y,
                     xlabel='Retransmissions', ylabel='End-to-end delay (ms)')  # ls='None')
    print('  ... LAT mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
def graph_pdr(df, out):
    global hlp
    """Graph end-to-end PDR."""
    sent = (df['TX'] == 1).sum()
    received = (df['RX'] == 1).sum()
    superfluous = ((df['TX'] == 0) & (df['RX'] == 1)).sum()
    x = [0]
    y = [hlp.ratio(sent, received - superfluous)]
    cpplot.plot_bar(df, 'tb_pdr', out + '/', x, y,
                    xlabel='Packet Drop Rate (%)', ylabel='End-to-end PDR (%)')
    print('  ... PDR mean: ' + str(np.mean(y)))


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    tmp_file = '/home/mike/Results/TMP'

    # Regex
    re_hb = '(?:(?P<heartbeat>\d+))\s<3+$'
    re_prefix = '^\[*(?:(?P<timestamp>.*))[\|\]]\s*'
    re_txrx = 'D:(?:\s*[ep:]+(?P<epoch>\d+)|\s+(?P<type>[FWDTRX-]+)\s+(?P<packet>.{1,16})|\s*[id:]+(?P<id>\d+)|\s*[s:]+(?P<src>\d+))*'

    # Cmd line args
    ap = argparse.ArgumentParser(prog='AtomicPy', description='Atomic Log Parser')
    ap.add_argument('--s', required=False, default=0, help='Save parsed logs')
    ap.add_argument('--c', required=False, default=0, help='Compare multiple saved results')
    ap.add_argument('--l', required=False, default=0, help='Load saved logs')
    ap.add_argument('--dir', required=True, help='Log directory')
    ap.add_argument('--title', required=False, default='ATM', help='Results title')
    ap.add_argument('--out', required=False, default='/home/mike/Results', help='Output folder')
    args = ap.parse_args()

    lp = LOG_PARSER()
    hlp = HELPER()

    out = args.out + '/' + args.title

    print(out)

    if args.l:
        print('.......... Load Results from ' + out)
        results = pickle.load(open(out + '/' + args.title + '_results.pkl', 'rb'))
    else:
        # Regex
        regex = re_prefix + re_txrx
        print('.......... Parse Data')
        df = lp.parse_logs(args.dir, None, tmp_file, regex)

        print('.......... Format Data')
        df.set_index('timestamp', inplace=True, drop=False)
        df.sort_index(inplace=True)
        # Remove outliers due to id screwing up in transmission
        # print("> Len:" + str(len(df.id)) + " Min:" + str(df.id.min()) + " Max:" + str(df.id.max()))
        # u_ids = df.id.unique()
        # u_ids = hlp.reject_outliers(u_ids)
        # df = df[df.id.isin(u_ids)]
        # u_ids = df.id.unique()
        # remove the nth ids because of logging delay and corrupt ids
        # df = df[df['id'] > np.partition(u_ids.flatten(), 5)[5]]  # min
        # df = df[df['id'] < np.partition(u_ids.flatten(), -5)[-5]]  # max
        print("> Len:" + str(len(df.id)) + " Min:" + str(df.id.min()) + " Max:" + str(df.id.max()))
        print('.......... Generate Results')
        results = generate_results(df)
        print(results)
        if args.s:
            print('.......... Save Results in ' + out)
            os.makedirs(out, exist_ok=True)
            pickle.dump(results, open(out + '/' + args.title + '_results.pkl', 'wb+'))

    print_results(results)

    print('.......... Pickle Results')
    sent = (results['TX'] == 1).sum()
    received = (results['RX'] == 1).sum()
    superfluous = ((results['TX'] == 0) & (results['RX'] == 1)).sum()
    pdr_mean = hlp.ratio(sent, received - superfluous)
    pickle.dump(pdr_mean, open(out + '/' + args.title + '_pdr.pkl', 'wb'))
    lat_mean = results.lat.mean()
    pickle.dump(lat_mean, open(out + '/' + args.title + '_lat.pkl', 'wb'))

    print('.......... Graph Results to ' + out)
    graph_latency(results, out)
    graph_pdr(results, out)

    if args.c:
        lat_0 = pickle.load(open(args.out + '/0/' + '0_lat.pkl', 'rb'))
        pdr_0 = pickle.load(open(args.out + '/0/' + '0_pdr.pkl', 'rb'))
        lat_25 = pickle.load(open(args.out + '/25/' + '25_lat.pkl', 'rb'))
        pdr_25 = pickle.load(open(args.out + '/25/' + '25_pdr.pkl', 'rb'))
        lat_50 = pickle.load(open(args.out + '/50/' + '50_lat.pkl', 'rb'))
        pdr_50 = pickle.load(open(args.out + '/50/' + '50_pdr.pkl', 'rb'))
        lat_75 = pickle.load(open(args.out + '/75/' + '75_lat.pkl', 'rb'))
        pdr_75 = pickle.load(open(args.out + '/75/' + '75_pdr.pkl', 'rb'))

        root = '/home/mike/Results/NIGHT_OUT/'
        x = ['0', '25', '50', '75']
        y = [pdr_0, pdr_25, pdr_50, pdr_75]
        # cpplot.plot_bar(results, 'tb_pdr', root, x, y,
        #                 xlabel='Packet Drop Rate (%)', ylabel='End-to-end PDR (%)')

        fig, ax1 = plt.subplots(figsize=(10, 6))
        # constants
        width = 0.35  # the width of the bars
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']
        ind = np.arange(len(x))
        ax1.bar(x=ind, height=y, width=width, color=color)
        xticks = np.arange(min(ind), max(ind)+1, 1.0)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(x)

        ax2 = ax1.twinx()
        y2 = [lat_0, lat_25, lat_50, lat_75]
        ax2.errorbar(xticks, y2, None, color='red', marker='D', ms=10, ls='--', lw=4)
        ax2.set_yticks(np.arange(0, max(y2) + 100, step=100))

        ax2.set_ylim([0, max(y2) + 100])
        # set axis' labels
        ax1.set_xlabel('Injected Rx Drop Rate (%)')
        ax2.set_ylabel('End-to-end Delay (ms)')

        ax1.set_ylabel('End-to-end PDR (%)')
        ax2.legend(['Mean Latency'], loc='lower right')
        fig.set_tight_layout(True)
        print('.......... Save to ' + root + 'fig_pdr_lat.pdf')
        fig.savefig(root + 'fig_pdr_lat.pdf', bbox_inches="tight")

        plt.show()
