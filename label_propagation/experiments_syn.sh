#!/bin/bash
# -*- coding: utf-8 -*-
#

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=20

# Make sure the program is up to date
make > /dev/null


function run {
    B=$2
    p=$3
    if [ -z "$4" ]; then
        q=0
    else
        q=$4
    fi
    for n in `seq 1 1`; do
        identifier=_${n}00000_20_${n}000_${mu}
        D=${PREFIX}/normalized_network${identifier}.dat
        C=${PREFIX}/normalized_community${identifier}.dat
        for x in `seq 1`; do
            echo -e "`date`\t$1$4\tn_LFR${identifier}\t${x}"
            bin/label_propagation $p $q $D 10 $B $C > /dev/null 2>> result_lfr.csv
        done
    done

    for n in `seq 1 1`; do
        identifier=_${n}00000_20_${n}000_${mu}
        D=${PREFIX}/randomized_network${identifier}.dat
        C=${PREFIX}/randomized_community${identifier}.dat
        for x in `seq 1`; do
            echo -e "`date`\t$1$4\tr_LFR${identifier}\t${x}"
            bin/label_propagation $p $q $D 10 $B $C > /dev/null 2>> result_lfr.csv
        done
    done
}


mu=0.2
PREFIX=../../graph_clustering_orig/lfr/binary_networks/datasets
DPP_IMPLS=(AsyncDPP SyncDPP HybridDPP InCoreLI InCoreDPP)
LFHT_IMPLS=(AsyncLFHT SyncLFHT HybridLFHT InCoreLFHT)

for (( i=0; i<${#DPP_IMPLS[@]}; i++ )); do
    impl=${DPP_IMPLS[$i]}
    if [ $i -eq 4 ]; then
        p=0
        B=30  # For in-core execution
    else
        p=$i
        B=20
    fi
    run $impl $B $p
done


for (( i=0; i<${#LFHT_IMPLS[@]}; i++ )); do
    impl=${LFHT_IMPLS[$i]}
    p=$(($i+4))
    if [ $i -eq 3 ]; then
        B=30  # For in-core execution
    else
        B=20
    fi
    for q in 0 1 2 3; do
        run $impl $B $p $q
    done
done
