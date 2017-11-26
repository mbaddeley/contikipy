#!/usr/bin/env python2.7
import subprocess
import argparse
import itertools as it
import os
import re
import shutil  # for copying files
import numpy as np
import matplotlib as plt
import pandas as pd
import collections
import yaml
import sys

import cplogparser as lp
import cpcsc as csc
import cpconfig as config

# import yaml config
cfg = yaml.load(open("config.yaml", 'r'))


# ----------------------------------------------------------------------------#
def main():
    # fetch arguments
    ap = argparse.ArgumentParser(prog='ContikiPy',
                                 description='Cooja simulation runner and '
                                             'Contiki log parser')
    ap.add_argument('--path', required=False, default=cfg['path'],
                    help='Absolute path to contiki')
    ap.add_argument('--target', required=False, default=cfg['target'],
                    help='Contiki platform TARGET')
    ap.add_argument('--log', required=False, default=cfg['log'],
                    help='Relative path to contiki log')
    ap.add_argument('--out', required=False, default=cfg['out'],
                    help='Absolute path to output folder')
    ap.add_argument('--wd', required=False, default=cfg['wd'],
                    help='(Relative) working directory for code + csc')
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
        simulations = [{'desc': '', 'makeargs': str(args.makeargs)}]
        analysis = None

    simlog = None
    print '**** Running through ' + str(len(simulations)) + ' simulations'
    for sim in simulations:
        print '> SIM: ',
        print sim
        # generate a simulation description
        desc = sim['desc']
        makeargs = sim['makeargs']
        plot_config = sim['plot']
        # Print some information about this simulation
        info = 'Running simulation: {0}'.format(desc)
        print '=' * len(info)
        print info
        print '=' * len(info)
        # HACK: replace remove list brackets
        makeargs = makeargs.replace('[', '').replace(']', '')
        print makeargs
        # make a note of our intended sim directory
        simdir = args.out + "/" + desc
        # run a cooja simulation with these sim settings
        if int(args.runcooja):
            if 'csc' in sim:
                csc = sim['csc']
            else:
                csc = args.csc
            simlog = run(args.path,
                         args.target,
                         args.log,
                         args.wd,
                         csc,
                         simdir,
                         makeargs,
                         desc)
        # generate results by parsing the cooja log
        if int(args.parse) and desc is not None:
            parse(simlog, simdir, desc, 'cooja', plot_config)

    # analyze the generated results
    if int(args.comp) and analysis is not None:
        print '**** Analyzing (comparing) results...'
        lp.compare_results(args.out, analysis['sims'], analysis['plots'])


# ----------------------------------------------------------------------------#
def parse(log, directory, desc, logfmt, plot_config):
    print '**** Parse log and gererate results in: ' + directory
    if log is None:
        log = directory + "/" + desc + ".log"
    lp.generate_results(log, directory + '/', logfmt, desc, plot_config)


# ----------------------------------------------------------------------------#
def run(contiki, target, log, wd, csc, outdir, args, desc):
    print '**** Clean and make: ' + contiki + wd + "/" + csc
    contikilog = contiki + log
    print '> Clean ' + contiki + wd
    clean(contiki + wd, target)
    # print '> Make ' + contiki + wd
    # make(contiki + wd, args)
    # Run the scenario in cooja with -nogui
    print '**** Create simulation directory'
    # Create a new folder for this scenario
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print '**** Running simulation ' + desc
    run_cooja(contiki, contiki + wd + '/' + csc, args)
    print '**** Copy log into simulation directory'
    # Copy contiki ouput log file and prefix the desc
    simlog = outdir + "/" + desc + '.log'
    shutil.copyfile(contikilog, simlog)

    return simlog


# ----------------------------------------------------------------------------#
def run_cooja(contiki, sim, args):
    # java = 'java -mx512m -jar'
    # cooja_jar = contiki + '/tools/cooja/dist/cooja.jar'
    # nogui = '-nogui=' + sim
    # contiki = '-contiki=' + contiki
    # cmd = args + ' ' + java + ' ' + cooja_jar + ' ' + nogui + ' ' + contiki
    ant = 'ant run_nogui'
    antbuild = '-file ' + contiki + '/tools/cooja/'
    antnogui = '-Dargs=' + sim
    cmd = args + ' ' + ant + ' ' + antbuild + ' ' + antnogui
    print '> ' + cmd
    subprocess.call(cmd, shell=True)


# ----------------------------------------------------------------------------#
def clean(path, target):
    # make clean in case we have new cmd line arguments
    subprocess.call('make clean TARGET=' + target + ' ' +
                    ' -C ' + path + '/controller',
                    shell=True)
    subprocess.call('make clean TARGET=' + target + ' ' +
                    ' -C ' + path + '/node',
                    shell=True)


# ----------------------------------------------------------------------------#
def make(path, args, target):
    print '> args: ' + args
    subprocess.call('make TARGET=' + target + ' ' + args +
                    ' -C ' + path + '/controller', shell=True)
    subprocess.call('make TARGET=' + target + ' ' + args +
                    ' -C ' + path + '/node',
                    shell=True)


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    # config = {}
    # execfile("mpy.conf", config)
    # print config["bitrate"]
    main()
