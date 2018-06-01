#!/bin/bash
# -*- coding: utf-8 -*-

. common.sh

result=output/result_p100_mg_ic.csv

# *-ooc
for i in 0 1; do
    for j in 2; do
        run_on_large $i $j
    done
done

# mg-ooc
run_on_large 1 4

# mg-ic
run_on_large 1 5
