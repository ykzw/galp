#!/bin/bash
# -*- coding: utf-8 -*-

. common.sh

result=output/result_p100_mid.csv

# *-ic, *-ooc
for i in 0 1; do
    for j in 0 2; do
        run_on_middle $i $j
    done
done
