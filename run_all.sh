#!/bin/bash
for FILE in ./config-atomic-v-usdn-*.yaml; do
  ./contikipy.py --conf="${FILE}" --runcooja=1
done
