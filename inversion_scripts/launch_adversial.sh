#!/bin/bash

for beam in `seq 10 10 50`; do
    for max_steps in `seq 10 10 50`; do
        sleep 2
        sbatch analyse_adversial_decoding.sh $beam $max_steps
    done
done
