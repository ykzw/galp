#!/bin/bash
# -*- coding: utf-8 -*-

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=40

repeat=5
mkdir -p output
result=output/result_p100.csv

program=../label_propagation/bin/galp
impls=(dpp-wop dpp-no dpp-o dpp-h dpp-li lfht-wop lfht-no lfht-o lfht-h mg-ooc mg-ic)

data_path=../datasets
small=$(echo "amazon dblp youtube" | xargs -n1 printf "normalized_com-%s.ungraph.txt ")
middle="lj.bin orkut.bin"
large="friendster.bin uk.bin"

MAX_GPUS=8

function run {
    index=$(( $1 * 5 + $2 ))
    impl=${impls[$index]}
    # Vary buffer sizes or not
    if [ $2 -eq 1 ] || [ $2 -eq 2 ] || [ $2 -eq 3 ] || [ $index -eq 9 ]; then
        bseq=`seq 21 28`
    else
        bseq=0
    fi

    # Vary the number of GPUs or not
    if [ $index -eq 9 ] || [ $index -eq 10 ]; then
        gseq=`seq 1 ${MAX_GPUS}`
    else
        gseq=1
    fi

    dataset_name=$(echo $d | sed -r "s/normalized_com-(.+).ungraph.txt/\1/" | sed "s/.bin//")

    for ngpus in $gseq; do
        for bs in $bseq; do
            for x in `seq ${repeat}`; do
                echo -e "`date`\t${impl}_b${bs}_g${ngpus}\t${dataset_name}\t${x}/${repeat}" >> log
                $program -b ${bs} -g ${ngpus} $1 $2 $3 2>&1 >/dev/null |
                    sed -r "s/normalized_com-(.+).ungraph.txt/\1/" |
                    sed "s/.bin//" >> $result
                if [ ${PIPESTATUS[0]} -eq 1 ]; then
                    break
                fi
            done
        done
    done
}


function run_on_datasets {
    for d in $3; do
        dataset=${data_path}/$d
        run $1 $2 $dataset
    done
}

function run_on_small {
    run_on_datasets $1 $2 "$small"
}

function run_on_middle {
    run_on_datasets $1 $2 "$middle"
}

function run_on_large {
    run_on_datasets $1 $2 "$large"
}
