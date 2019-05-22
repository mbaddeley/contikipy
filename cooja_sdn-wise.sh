#!/bin/bash
trap 'killallp' INT

killallp() {
    trap '' INT TERM     # ignore INT and TERM while shutting down
    echo -n "> Sleepy run_misses shutting down... "     # added double quotes
    kill -TERM 0         # fixed order, send TERM not INT
    wait
    echo "DONE"
}

killall tail

# $1 make args
# $2 simulation

RESULTSDIR="/home/mike/Results/SDN_WISE"
CONTIKIDIR="/home/mike/Repos/sdn-wise-contiki/sdn-wise"
COOJADIR="/home/mike/Repos/sdn-wise-contiki/contiki/tools/cooja/"
COOJALOG="$COOJADIR/build/COOJA.testlog"
SIMULATION="/home/mike/Repos/sdn-wise-contiki/sdn-wise/$1"
TARGET="sky"

copylogs() {
  echo -n "> Copy logs... "
  mkdir -p $RESULTS_DIR
  cp $COOJA_LOG/* $RESULTS_DIR
  echo "DONE"
}

compile() {
  echo "*** Compiling... $1 ***"
  gnome-terminal --tab --title=$2 -- bash -c "cd $CONTIKIDIR;make clean TARGET=$TARGET && make TARGET=$TARGET $1;echo \"***FINISHED*** $1\";exec bash"
  sleep 5
}

run_cooja() {
  echo "*** Run Cooja... $1 ***"
  gnome-terminal --tab --title=$2 -- bash -c "cd $COOJADIR;ant run_nogui -file $COOJADIR -Dargs=$SIMULATION;echo \"***FINISHED*** $SIMULATION\";exec bash"
}

echo START
ARGS="$2"
compile "SINK=1 $ARGS" sink
mv $CONTIKIDIR/sdn-wise.$TARGET $CONTIKIDIR/sink.$TARGET
compile "SINK=0 $ARGS" node
mv $CONTIKIDIR/sdn-wise.$TARGET $CONTIKIDIR/sink.$TARGET

run_cooja

echo FINISHED!
wait < <(jobs -p)
