#!/bin/bash
# -*- coding: utf-8 -*-

. common.sh

result=output/result_lfr.csv

# *-ic
for i in 5 10; do
    for m in 0 1; do
        p=0
        for j in `seq 1 10`; do
            dataset=${data_path}/${i}_${j}/normalized_network.dat
            testfile=${data_path}/${i}_${j}/normalized_community.dat
            echo -e "`date`\t${impl}_b${bs}_g${ngpus}\t${dataset_name}\t${x}/${repeat}" >> log
            $program $m $p $dataset $testfile 2>&1 >/dev/null |
                sed -r "s/normalized_//" >> $result
        done
    done
done

# *-o
for i in 5 10; do
    for m in 0 1; do
        p=2
        for b in `seq 20 22`; do
            for j in `seq 1 10`; do
                dataset=${data_path}/${i}_${j}/normalized_network.dat
                testfile=${data_path}/${i}_${j}/normalized_community.dat
                echo -e "`date`\t${impl}_b${bs}_g${ngpus}\t${dataset_name}\t${x}/${repeat}" >> log
                $program -b $b $m $p $dataset $testfile 2>&1 >/dev/null |
                    sed -r "s/normalized_//" >> $result
            done
        done
    done
done


# # Serial
# for i in 10; do
#     for j in `seq 1 10`; do
#         dataset=${data_path}/${i}_${j}/normalized_network.dat
#         testfile=${data_path}/${i}_${j}/normalized_community.dat
#         echo -e "`date`\t${impl}_b${bs}_g${ngpus}\t${dataset_name}\t${x}/${repeat}" >> log
#         ${program}_2 30 0 $dataset $testfile 2>&1 >/dev/null |
#             sed -r "s/normalized_//" >> $result
#     done
# done
