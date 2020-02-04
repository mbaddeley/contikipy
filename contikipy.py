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
import cpcsc

import pickle   # for saving data

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
    ap.add_argument('--l', required=False, default=0, help='Load saved logs')
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
    for sim in simulations:
        sim_desc, sim_type, makeargs, regex, plots = get_sim_config(sim)
        sim_dir = args.out + "/" + sim_desc + '/'
        if 'log' in sim:
            cooja_log = '/home/mike/Results/' + sim['log'] + '.log'
            sim_log = sim_dir + sim['log'] + '.log'
        else:
            cooja_log = args.contiki + "/tools/cooja/build/COOJA.testlog"
            sim_log = sim_dir + sim_desc + '.log'
        # Create a new folder for this scenario
        if not os.path.exists(sim_dir):
            print('**** Create simulation directory')
            os.makedirs(sim_dir)
# RUNCOOJA ----------------------------------------------------------- RUNCOOJA
        if int(args.runcooja):
            print('**** Run ' + str(len(simulations)) + ' simulations')
            title = sim['log'] if 'log' in sim else sim_desc
            runcooja(args, sim, sim_dir, makeargs, title)
            print('**** Copy cooja log into simulation directory')
            shutil.copyfile(cooja_log, sim_log)
# PARSE ----------------------------------------------------------------- PARSE
        if int(args.parse) and sim_desc is not None:
            print('**** Parse data from ' + args.conf + ' -> ' + sim_desc)
            logtype_re = cfg['logtypes']['cooja']
            if args.l:
                print('>  Load saved data ... ' + sim_dir + sim_desc + '_df.pkl')
                df_dict = pickle.load(open(sim_dir + sim_desc + '_df.pkl', 'rb'))
                parse(logtype_re, sim_log, sim_dir, sim_desc, sim_type, regex, plots, df_dict)
            else:
                parse(logtype_re, sim_log, sim_dir, sim_desc, sim_type, regex, plots, None)
# COMP ------------------------------------------------------------------- COMP
    if int(args.comp) and compare is not None:
        compare_args = compare['args'] if 'args' in compare else None
        print('**** Compare plots in dir: ' + args.out)
        print(compare)
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
    plots = sim['plot'] if 'plot' in sim else None

    return sim_desc, sim_type, makeargs, regex, plots


# ----------------------------------------------------------------------------#
def runcooja(args, sim, outdir, makeargs, title):
    """Run cooja."""
    try:
        # print some information about this simulation
        simstr = 'Simulation: ' + title
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
        # configure csc simulation
        cpcsc.set_simulation_title(wd + '/' + csc, title)
        print('**** Run simulation: ' + title)
        run(contiki, args.target, wd, csc, makeargs)
        # reset csc simulation
        cpcsc.set_simulation_title(wd + '/' + csc, csc.split('.csc')[0])
    except Exception:
        traceback.print_exc()
        sys.exit(0)


# ----------------------------------------------------------------------------#
def parse(logtype_re, sim_log, sim_dir, sim_desc, sim_type, pattern_types, plot_types, df_dict):
    """Parse the main log for each datatype."""
    global cfg
    # print(some information about what's being parsed
    simstr = '- Simulation: ' + sim_desc
    dirstr = '- Directory: ' + sim_dir
    plotstr = '- Plots {0}'.format(plot_types.keys())
    info = simstr + '\n' + dirstr + '\n' + plotstr
    info_len = len(max([simstr, dirstr, plotstr], key=len))
    print('-' * info_len)
    print(info)
    print('-' * info_len)

    if df_dict is None:
        df_dict = {}
        for p in pattern_types:
            df_dict.update({p: parse_regex(sim_log, logtype_re, p, sim_dir)})
        if bool(df_dict):
            # Do any YAML configured data processing
            if 'process' in cfg['formatters']:
                if cfg['formatters']['process'] is not None:
                    print('  > Process the data...')
                    df_dict = process_data(df_dict, cfg['formatters']['process'])
        # Save the data
        print('> Saving data as ... ' + sim_dir + sim_desc + '_df.pkl')
        pickle.dump(df_dict, open(sim_dir + sim_desc + '_df.pkl', 'wb+'))
    # else:
    #     print('**** Successfully loaded pickle data...')
    # """Generate plots."""
    print('**** Generate the following plots: [' + ', '.join(plot_types) + ']')
    lp.plot_data(sim_desc, sim_type, sim_dir, df_dict, plot_types)


# ----------------------------------------------------------------------------#
def parse_regex(sim_log, logtype_re, pattern_type, sim_dir):
    """Parse log for data regex."""
    try:
        for p in cfg['formatters']['patterns']:
            if pattern_type in p['type']:
                regex = logtype_re + p['regex']
                pattern = lp.scrape_data(p['type'], sim_log, sim_dir, regex)
                if not (pattern is None):
                    return pattern
                else:
                    raise Exception("No pattern!")
    except Exception:
        traceback.print_exc()
        sys.exit(0)


# ----------------------------------------------------------------------------#
def process_data(data_dict, process_list):
    """Process the dataframes."""
    # Check to see if we have a processing task for each data df
    for k, v in cfg['formatters']['process'].items():
        if k in data_dict:
            for p in v:
                df = data_dict[k]
                if 'merge' in p.keys():
                    m = p['merge']
                    print('  ... Merge ' + k + ': ' + str(m))
                    how = m['how'] if 'how' in m else 'inner'
                    df = df.merge(data_dict[m['df']], how,
                                  left_on=m['left_on'], right_on=m['right_on'])
                if 'filter' in p.keys():
                    f = p['filter']
                    print('  ... Filter ' + k + ': ' + str(f))
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
def run(contiki, target, wd, csc, args):
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
    if args is None:
        args = ""
    ant = 'ant run_nogui'
    antbuild = '-file ' + contiki + '/tools/cooja/'
    antnogui = '-Dargs=' + csc
    cmd = ant + ' ' + antbuild + ' ' + antnogui
    print('> ' + cmd)
    subprocess.call(cmd, shell=True)


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    # config = {}
    # execfile("mpy.conf", config)
    # print(config["bitrate"]
    main()
