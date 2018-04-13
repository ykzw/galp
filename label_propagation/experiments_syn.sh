#!/bin/bash
# -*- coding: utf-8 -*-
#

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=20

# Make sure the program is up to date
make > /dev/null


mu=0.2
PREFIX=../../graph_clustering_orig/lfr/binary_networks/datasets
IMPLS=(Async Sync LI Hybrid AsyncLFHT)

for p in 4; do
    impl=${IMPLS[$p]}
    if [ $p -eq 1 ]; then
        p=0
        B=20
    elif [ $p -eq 4 ]; then
        B=20
    else
        B=30
    fi

    for n in `seq 1 10`; do
        identifier=_${n}00000_20_${n}000_${mu}
        D=${PREFIX}/normalized_network${identifier}.dat
        C=${PREFIX}/normalized_community${identifier}.dat
        for i in `seq 1`; do
            echo -e "`date`\t${impl}\tn_LFR${identifier}\t${i}"
            bin/label_propagation $p $D 10 $B $C > /dev/null 2>> result_lfr.csv
        done
    done

    for n in `seq 1 10`; do
        identifier=_${n}00000_20_${n}000_${mu}
        D=${PREFIX}/randomized_network${identifier}.dat
        C=${PREFIX}/randomized_community${identifier}.dat
        for i in `seq 1`; do
            echo -e "`date`\t${impl}\tr_LFR${identifier}\t${i}"
            bin/label_propagation $p $D 10 $B $C > /dev/null 2>> result_lfr.csv
        done
    done
done
