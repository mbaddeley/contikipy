#!/usr/bin/python
"""Contiki log parser."""
import argparse
import os
import shutil  # for copying files
import subprocess
import traceback
import sys

import yaml

import cpconfig as config
import cplogparser as lp
import cpcomp

# import yaml config
cfg = yaml.load(open("config-atomic-v-usdn.yaml", 'r'))


# ----------------------------------------------------------------------------#
def run_cooja(args, sim, outdir, makeargs, simname):
    """Run cooja."""
    try:
        # check for working directory in the sim
        if 'contiki' in sim:
            contiki = sim['contiki']
        elif args.contiki is not None:
            contiki = args.contiki
        else:
            raise Exception('ERROR: No path to contiki!')
        if 'wd' in sim:
            wd = sim['wd']
        elif args.wd is not None:
            wd = args.wd
        else:
            raise Exception('ERROR: No working directory!')
        # check for csc in the sim
        if 'csc' in sim:
            csc = sim['csc']
        elif args.csc is not None:
            csc = args.csc
        else:
            raise Exception('ERROR: No csc file!')
        simlog = run(contiki,
                     args.target,
                     args.log,
                     wd,
                     csc,
                     outdir,
                     makeargs,
                     simname)
        return simlog
    except Exception:
        traceback.print_exc()
        sys.exit(0)


# ----------------------------------------------------------------------------#
def parse(log, dir, simname, fmt, plots):
    """Parse the main log for each datatype."""
    if log is None:
        log = dir + simname + ".log"
    print('**** Parse log and gererate data logs in: ' + dir)
    logtype = (l for l in cfg['logtypes'] if l['type'] == fmt).next()
    df_dict = {}
    for d in cfg['formatters']['dictionary']:
        regex = logtype['fmt_re'] + d['regex']
        df = lp.scrape_data(d['type'], log, dir, fmt, regex)
        if df is not None:
            df_dict.update({d['type']: df})
    if bool(df_dict):
        print('**** Pickle the data...')
        lp.pickle_data(dir, df_dict)
    if plots is not None:
        print('**** Generate the following plots: [' + ' '.join(plots) + ']')
        lp.plot_data(simname, dir, df_dict, plots)


# ----------------------------------------------------------------------------#
def main():
    """Take in command line arguments and parses log accordingly."""
    # fetch arguments
    ap = argparse.ArgumentParser(prog='ContikiPy',
                                 description='Cooja simulation runner and '
                                             'Contiki log parser')
    ap.add_argument('--contiki', required=False, default=cfg['contiki'],
                    help='Absolute path to contiki')
    ap.add_argument('--target', required=False, default=cfg['target'],
                    help='Contiki platform TARGET')
    ap.add_argument('--log', required=False, default=cfg['log'],
                    help='Relative path to contiki log')
    ap.add_argument('--out', required=False, default=cfg['out'],
                    help='Absolute path to output folder')
    ap.add_argument('--wd', required=False, default=cfg['wd'],
                    help='(Relative) working directory for code + csc')
    ap.add_argument('--fmt', required=False, default=cfg['fmt'],
                    help='Cooja simulation file')
    ap.add_argument('--csc', required=False, default=cfg['csc'],
                    help='Cooja simulation file')
    ap.add_argument('--makeargs', required=False,
                    help='Makefile arguments')
    ap.add_argument('--runcooja', required=False, default=0,
                    help='Run the simulation')
    ap.add_argument('--parse', required=False, default=0,
                    help='Run the log parser')
    ap.add_argument('--comp', required=False, default=0,
                    help='Run the analyzer (compare sims)')
    args = ap.parse_args()

    if not args.makeargs:
        # get simulations configuration
        conf = config.Config()
        simulations = conf.simconfig()
        # get analysis configuration
        analysis = conf.analysisconfig()
    else:
        simulations = [{'simname': '', 'makeargs': str(args.makeargs)}]
        analysis = None

    simlog = None
    print('**** Run ' + str(len(simulations)) + ' simulations')
    for sim in simulations:
        # generate a simulation description
        simname = sim['desc']
        if 'makeargs' in sim and sim['makeargs'] is not None:
            makeargs = sim['makeargs']
        else:
            makeargs = None
        plot_config = sim['plot']
        # print(some information about this simulation
        info = 'Running simulation: {0}'.format(simname)
        print('=' * len(info))
        print(info)
        print('=' * len(info))
        # HACK: replace remove list brackets
        if makeargs is not None:
            makeargs = makeargs.replace('[', '').replace(']', '')
            print(makeargs)
        # make a note of our intended sim directory
        outdir = args.out + "/" + simname + "/"
        # run a cooja simulation with these sim settings
        if int(args.runcooja):
            simlog = run_cooja(args, sim, outdir, makeargs, simname)
        # generate results by parsing the cooja log
        if int(args.parse) and simname is not None:
            parse(simlog, outdir, simname, args.fmt, plot_config)

    # analyze the generated results
    if int(args.comp) and analysis is not None:
        print('**** Compare plots in dir: ' + args.out)
        cpcomp.compare(args.out, analysis['sims'], analysis['plots'])


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
    print(target_str)
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
    subprocess.call('make ' + target_str + ' ' + args +
                    ' -C ' + path, shell=True)


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
                print('> Make ' + dir)
                make(dir, target, args)
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
    cmd = args + ' ' + ant + ' ' + antbuild + ' ' + antnogui
    print('> ' + cmd)
    subprocess.call(cmd, shell=True)
    print('**** Copy contiki log into simulation directory')
    # Copy contiki ouput log file and prefix the simname
    simlog = outdir + simname + '.log'
    contikilog = contiki + log
    shutil.copyfile(contikilog, simlog)

    return simlog


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    # config = {}
    # execfile("mpy.conf", config)
    # print(config["bitrate"]
    main()
