#!/bin/bash

#register=$1
#for depth in `seq 20 10 50`; do
#    for num_sent in 1 2 3; do
#        sleep 1
#        sbatch v2t.sh $num_sent $depth $register
#    done
#done


register=$1
for depth in `seq 30 10 50`; do
    sleep 1
    sbatch v2t.sh 0 $depth $register
done