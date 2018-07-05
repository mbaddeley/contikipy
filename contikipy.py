#!/usr/bin/python
"""Contiki log parser."""
import argparse
import os
import shutil  # for copying files
import subprocess
import traceback
import sys

import yaml

import cpconfig
import cplogparser as lp
import cpcomp

# import yaml config
cfg = None


# ----------------------------------------------------------------------------#
def run_cooja(args, sim, outdir, makeargs, simname):
    """Run cooja."""
    try:
        # check for working directory in the sim
        if 'contiki' in sim:
            contiki = sim['contiki']
        elif 'contiki' in args and args.contiki:
            contiki = args.contiki
        else:
            raise Exception('ERROR: No path to contiki!')
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
def parse(log, dir, simname, simtype, fmt, regex_list, plots):
    """Parse the main log for each datatype."""
    global cfg
    if log is None:
        log = dir + simname + ".log"
    # print(some information about what's being parsed
    simstr = '- Simulation: ' + simname
    dirstr = '- Directory: ' + dir
    plotstr = '- Plots {0}'.format(plots)
    info = simstr + '\n' + dirstr + '\n' + plotstr
    info_len = len(max([simstr, dirstr, plotstr], key=len))
    print('-' * info_len)
    print(info)
    print('-' * info_len)
    print('**** Parse log and gererate data logs in: ' + dir)
    logtype = (l for l in cfg['logtypes'] if l['type'] == fmt).next()
    df_dict = {}
    for d in cfg['formatters']['dictionary']:
        if regex_list is None or d['type'] in regex_list:
            regex = logtype['fmt_re'] + d['regex']
            df = lp.scrape_data(d['type'], log, dir, fmt, regex)
            if df is not None:
                df_dict.update({d['type']: df})
    if bool(df_dict):
        print('**** Pickle the data...')
        lp.pickle_data(dir, df_dict)
    if plots is not None:
        print('**** Generate the following plots: [' + ' '.join(plots) + ']')
        lp.plot_data(simname, simtype, dir, df_dict, plots)


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
    ap.add_argument('--log', required=False,
                    help='Relative path to contiki log')
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
    args.log = cfg['log'] if not args.log else args.log
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

    simlog = None
    print('**** Run ' + str(len(simulations)) + ' simulations')
    for sim in simulations:
        # get simulation description
        sim_desc = sim['desc']
        # get simulation type
        sim_type = sim['type']
        # get makeargs
        if 'makeargs' in sim and sim['makeargs'] is not None:
            makeargs = sim['makeargs']
        else:
            makeargs = None
        # get regex
        regex = sim['regex'] if 'regex' in sim else None

        plot_config = sim['plot']
        # print(some information about this simulation
        info = 'Running simulation: {0}'.format(sim_desc)
        print('=' * len(info))
        print(info)
        print('=' * len(info))

        if makeargs is not None:
            # HACK: replace remove list brackets
            makeargs = makeargs.replace('[', '').replace(']', '')
            print(makeargs)

        # make a note of our intended sim directory
        sim_dir = args.out + "/" + sim_desc + "/"
        # run a cooja simulation with these sim settings
        if int(args.runcooja):
            simlog = run_cooja(args, sim, sim_dir, makeargs, sim_desc)
        # generate results by parsing the cooja log
        if int(args.parse) and sim_desc is not None:
            parse(simlog, sim_dir, sim_desc, sim_type,
                  cfg['fmt'], regex, plot_config)

    # analyze the generated results
    if int(args.comp) and compare is not None:
        print('**** Compare plots in dir: ' + args.out)
        cpcomp.compare(args.out,  # directory
                       compare['sims'],
                       # {compare['sims']: compare['labels']},  # sims + labels
                       compare['plots'])  # plots to compare


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
    print("-------------> " + target_str + " " + args)
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
    # TODO: can we instruct cooja to write to a new log each time?
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
