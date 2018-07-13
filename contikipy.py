#!/usr/bin/python
"""Contiki log parser."""
import argparse
import os
import shutil  # for copying files
import subprocess
import traceback
import sys

import yaml
import pandas as pd

import cpconfig
import cplogparser as lp
import cpcomp
# import cpcsc

from pprint import pprint

# yaml config
cfg = None


# ----------------------------------------------------------------------------#
def main():
    """Take in command line arguments and parses log accordingly."""
    global cfg
    # fetch arguments
    ap = argparse.ArgumentParser(prog='ContikiPy',
                                 description='Cooja simulation runner and '
                                             'Contiki log parser')
    ap.add_argument('--conf', required=True,
                    help='YAML config file to use')
    ap.add_argument('--runcooja', required=False, default=0,
                    help='Run the simulation')
    ap.add_argument('--parse', required=False, default=0,
                    help='Run the log parser')
    ap.add_argument('--comp', required=False, default=0,
                    help='Compare plots')
    ap.add_argument('--contiki', required=False, default=0,
                    help='Absolute path to contiki folder')
    ap.add_argument('--out', required=False,
                    help='Absolute path to output folder')
    ap.add_argument('--wd', required=False,
                    help='(Relative) working directory for code + csc')
    ap.add_argument('--csc', required=False,
                    help='Cooja simulation file')
    ap.add_argument('--target', required=False,
                    help='Contiki platform TARGET')
    ap.add_argument('--makeargs', required=False,
                    help='Makefile arguments')
    args = ap.parse_args()
    cfg = yaml.load(open(args.conf, 'r'))

    args.contiki = cfg['contiki'] if not args.contiki else args.contiki
    args.out = cfg['out'] if not args.out else args.out
    args.wd = cfg['wd'] if not args.wd else args.wd
    args.csc = cfg['csc'] if not args.csc else args.csc
    args.target = cfg['target'] if not args.target else args.target

    if not args.makeargs:
        # get simulations configuration
        conf = cpconfig.Config(cfg)
        simulations = conf.simconfig()
        # get compare configuration
        compare = conf.compareconfig()
    else:
        simulations = [{'simname': '', 'makeargs': str(args.makeargs)}]
        compare = None

    # get simulation config
    print('**** Run ' + str(len(simulations)) + ' simulations')
    for sim in simulations:
        sim_desc, sim_type, makeargs, regex, plots = get_sim_config(sim)
        sim_dir = args.out + "/" + sim_desc + "/"
        # check for cooja log in the sim
        if 'log' in sim:
            sim_log = sim['log']
            # cpcsc.set_simulation_title(log)
        else:
            sim_log = args.contiki + "/tools/cooja/build/COOJA.testlog"
# RUNCOOJA ----------------------------------------------------------- RUNCOOJA
        if int(args.runcooja):
            runcooja(args, sim, sim_dir, makeargs, sim_desc, sim_log)
# PARSE ----------------------------------------------------------------- PARSE
        if int(args.parse) and sim_desc is not None:
            logtype_re = cfg['logtypes']['cooja']
            parse(logtype_re, sim_log, sim_dir, sim_desc, sim_type,
                  regex, plots)
# COMP ------------------------------------------------------------------- COMP
    if int(args.comp) and compare is not None:
        compare_args = compare['args'] if 'args' in compare else None
        print('**** Compare plots in dir: ' + args.out)
        cpcomp.compare(args.out,  # directory
                       compare['sims'],
                       compare['plots'],
                       compare_args)  # plots to compare


# ----------------------------------------------------------------------------#
def get_sim_config(sim):
    """Get the simulation configuration."""
    sim_desc = sim['desc']
    sim_type = sim['type']
    makeargs = sim['makeargs'] if 'makeargs' in sim else None
    regex = sim['regex'] if 'regex' in sim else None
    plots = sim['plot']

    return sim_desc, sim_type, makeargs, regex, plots


# ----------------------------------------------------------------------------#
def runcooja(args, sim, outdir, makeargs, sim_desc, sim_log):
    """Run cooja."""
    try:
        # print some information about this simulation
        simstr = 'Simulation: ' + sim_desc
        dirstr = 'Directory: ' + outdir
        info = simstr + '\n' + dirstr
        print('-' * len(info))
        print(info)
        print('-' * len(info))
        # check for contiki directory in the sim
        if 'contiki' in sim:
            contiki = sim['contiki']
        elif 'contiki' in args and args.contiki:
            contiki = args.contiki
        else:
            raise Exception('ERROR: No path to contiki!')
        # check for working directory in the sim
        if 'wd' in sim:
            wd = sim['wd']
        elif 'wd' in args and args.wd:
            wd = args.wd
        else:
            raise Exception('ERROR: No working directory!')
        # check for csc in the sim
        if 'csc' in sim:
            csc = sim['csc']
        elif 'csc' in args and args.csc:
            csc = args.csc
        else:
            raise Exception('ERROR: No csc file!')
        run(contiki, args.target, sim_log, wd, csc, outdir, makeargs, sim_desc)
    except Exception:
        traceback.print_exc()
        sys.exit(0)


# ----------------------------------------------------------------------------#
def parse(logtype_re, sim_log, sim_dir, sim_desc, sim_type,
          pattern_types, plot_types):
    """Parse the main log for each datatype."""
    global cfg
    # print(some information about what's being parsed
    simstr = '- Simulation: ' + sim_desc
    dirstr = '- Directory: ' + sim_dir
    plotstr = '- Plots {0}'.format(plot_types)
    info = simstr + '\n' + dirstr + '\n' + plotstr
    info_len = len(max([simstr, dirstr, plotstr], key=len))
    print('-' * info_len)
    print(info)
    print('-' * info_len)
    print('**** Parse log and gererate data logs in: ' + sim_dir)
    df_dict = {}
    for p in pattern_types:
        df_dict.update({p: parse_regex(sim_log, logtype_re, p, sim_dir)})
    # Save
    if bool(df_dict):
        print('**** Process the data...')
        df_dict = process_data(df_dict, cfg['formatters']['process'])
        print('**** Pickle the data...')
        lp.pickle_data(sim_dir, df_dict)
    # """Generate plots."""
    print('**** Generate the following plots: [' + ' '.join(plot_types) + ']')
    lp.plot_data(sim_desc, sim_type, sim_dir, df_dict, plot_types)


# ----------------------------------------------------------------------------#
def parse_regex(sim_log, logtype_re, pattern_type, sim_dir):
    """Parse log for data regex."""
    for p in cfg['formatters']['patterns']:
        if pattern_type in p['type']:
            regex = logtype_re + p['regex']
            return lp.scrape_data(p['type'], sim_log, sim_dir, regex)


# ----------------------------------------------------------------------------#
def process_data(data_dict, process_list):
    """Process the dataframes."""
    # Check to see if we have a processing task for each data df
    for k, v in cfg['formatters']['process'].items():
        if k in data_dict:
            df = data_dict[k]
            if 'merge' in v:
                m = v['merge']
                pprint(df)
                df = df.merge(data_dict[m['df']], left_on=m['left_on'],
                              right_on=m['right_on'])
            if 'filter' in v:
                f = v['filter']
                df = df[(df[f['col']] >= f['min'])
                        & (df[f['col']] <= f['max'])]
            data_dict[k] = df

    return data_dict


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
def clean(path, target):
    """Run contiki make clean."""
    # make clean in case we have new cmd line arguments
    if target is None:
        target_str = ""
    else:
        target_str = 'TARGET=' + target
    subprocess.call('make clean ' + target_str + ' -C ' + path, shell=True)


# ----------------------------------------------------------------------------#
def make(path, target, args=None):
    """Run contiki make."""
    if args is None:
        args = ""
    print('> args: ' + args)
    if target is None:
        target_str = ""
    else:
        target_str = 'TARGET=' + target
    subprocess.call('make ' + target_str + ' ' + args
                    + ' -C ' + path, shell=True)


# ----------------------------------------------------------------------------#
def run(contiki, target, log, wd, csc, outdir, args, simname):
    """Clean, make, and run cooja."""
    csc = wd + "/" + csc
    print('**** Clean and make: ' + csc)
    # Find makefiles in the working directory and clean + make
    for root, dirs, files in walklevel(wd):
        for dir in dirs:
            dir = wd + '/' + dir
            if 'Makefile' in os.listdir(dir):
                print('> Clean ' + dir)
                clean(dir, target)
                print('> Make ' + args)
                make(dir, target, args)

        print('> Clean ' + root)
        clean(root, target)
        print('> Make ' + root)
        make(root, target, args)
    # Run the scenario in cooja with -nogui
    print('**** Create simulation directory')
    # Create a new folder for this scenario
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('**** Running simulation ' + simname)
    if args is None:
        args = ""
    ant = 'ant run_nogui'
    antbuild = '-file ' + contiki + '/tools/cooja/'
    antnogui = '-Dargs=' + csc
    cmd = ant + ' ' + antbuild + ' ' + antnogui
    print('> ' + cmd)
    subprocess.call(cmd, shell=True)
    print('**** Copy contiki log into simulation directory')
    # Copy contiki ouput log file and prefix the simname
    simlog = outdir + simname + '.log'
    shutil.copyfile(log, simlog)

    return simlog


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    # config = {}
    # execfile("mpy.conf", config)
    # print(config["bitrate"]
    main()
