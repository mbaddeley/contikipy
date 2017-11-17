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

import cplogparser as lp
import cpcsc as csc
import cpconfig as config

# import yaml config
cfg = yaml.load(open("config.yaml", 'r'))


# ----------------------------------------------------------------------------#
def main():
    # fetch arguments
    ap = argparse.ArgumentParser(description='Simulation log parser')
    ap.add_argument('--contiki', required=False, default=cfg['contiki'],
                    help='Absolute path to contiki')
    ap.add_argument('--log', required=False, default=cfg['log'],
                    help='Relative path to contiki log')
    ap.add_argument('--out', required=False, default=cfg['out'],
                    help='Absolute path to output folder')
    ap.add_argument('--runcooja', required=False, default=0,
                    help='Run the simulation')
    ap.add_argument('--parse', required=False, default=0,
                    help='Run the log parser')
    ap.add_argument('--analyze', required=False, default=0,
                    help='Run the analyzer')
    ap.add_argument('--wd', required=False, default=cfg['wd'],
                    help='(Relative) working directory for example code + csc')
    ap.add_argument('--csc', required=False, default=cfg['csc'],
                    help='Cooja simulation file')
    ap.add_argument('--makeargs', required=False,
                    help='Optional makefile arguments')
    args = ap.parse_args()

    contiki = args.contiki

    # get makefile arguments
    if not args.makeargs:
        # Single application scenarios
        conf = config.SingleAppConfig()
        argstrings = conf.args()
        # Multi application scenarios
        # conf = config.MultiAppConfig()
        # argstrings = conf.args()
    else:
        argstrings = [str(args.makeargs)]

    print '**** Compiling argstrings'
    print argstrings

    # regex for simulation description
    sim_pattern = '(?:\w*RPL_MODE=(?P<RPL>\w+)' \
                  '|\s+SDN=(?P<SDN>[\d,]+)' \
                  '|\s+CBR=(?P<CBR>[\d,]+)' \
                  '|\s+VBR_MAX=\[(?P<MAX>[\d,]+)\]' \
                  '|\s+VBR_MIN=\[(?P<MIN>[\d,]+)\]' \
                  '|\s+NUM_APPS=(?P<APPS>[\d,]+)' \
                  '|\s+REROUTE=(?P<SCEN>[\d,]+)' \
                  '|\s+\w+=\S+)+.*?$'
    desc_re = re.compile(sim_pattern)

    print '**** Compiling simpattern'
    print sim_pattern

    sim_log = None
    for makeargs in argstrings:
        # generate a simulation description
        desc = ''
        m = desc_re.match(makeargs)
        if m:
            g = m.groupdict().iteritems()
            for k, v in g:
                if v is not None:
                    v = v.replace(',', '-')
                    desc += (k + '_' + v + '_')
            desc = desc[:-1]  # remove trailing '_'
            # Print some information about this simulation
            info = 'Running scenario: {0}'.format(desc)
            print '=' * len(info)
            print info
            print '=' * len(info)
            # HACK: replace remove list brackets
            makeargs = makeargs.replace('[', '').replace(']', '')
            # make a note of our intended sim directory
            simdir = args.out + "/" + desc
            # run a cooja simulation with these sim settings
            if int(args.runcooja):
                sim_log = run(contiki,
                              args.log,
                              args.wd,
                              args.csc,
                              simdir,
                              makeargs,
                              desc)

            # generate results by parsing the cooja log
            if int(args.parse) and desc is not None:
                parse(simdir, desc, 'cooja')

    # analyze the generated results
    if int(args.analyze):
        print '**** Analyzing results: '
        lp.analyze_results()


# ----------------------------------------------------------------------------#
def parse(directory, desc, logfmt):
    log = directory + '/' + desc + '.log'
    print '**** Parse log and gererate results in: ' + directory
    lp.generate_results(log, directory + '/', logfmt, desc)


# ----------------------------------------------------------------------------#
def run(contiki, log, wd, filename, outdir, args, desc):
    print '**** Clean and make: ' + contiki + wd + filename
    contikilog = contiki + log
    print '> Clean ' + contiki + wd
    clean(contiki + wd)
    # print '> Make ' + contiki + wd
    # make(contiki + wd, args)
    # Run the scenario in cooja with -nogui
    print '**** Running simulation ' + desc
    run_cooja(contiki, contiki + wd + '/' + filename, args)
    # Plot results
    print '**** Parse simulation logs'
    # Create a new folder for this scenario
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Copy contiki ouput log file and prefix the desc
    new_log = outdir + '.log'
    shutil.copyfile(contikilog, new_log)

    return new_log


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
def clean(path):
    # make clean in case we have new cmd line arguments
    subprocess.call('make clean TARGET=exp5438 '
                    ' -C ' + path + '/controller',
                    shell=True)
    subprocess.call('make clean TARGET=exp5438 '
                    ' -C ' + path + '/node',
                    shell=True)


# ----------------------------------------------------------------------------#
def make(path, args):
    print '> args: ' + args
    subprocess.call('make TARGET=exp5438 ' + args +
                    ' -C ' + path + '/controller',
                    shell=True)
    subprocess.call('make TARGET=exp5438 ' + args +
                    ' -C ' + path + '/node',
                    shell=True)


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    # config = {}
    # execfile("mpy.conf", config)
    # print config["bitrate"]
    main()
