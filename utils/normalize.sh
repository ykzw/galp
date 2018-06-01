#!/bin/bash
# -*- coding: utf-8 -*-


function normalize() {
    sed 's/\t/ /g' $1 | sed -e'1,4d' > tmp.txt
    python3 ../utils/normalize.py tmp.txt $2
    mv normalized_tmp.txt "normalized_$1"
}

cd ../datasets

normalize com-amazon.ungraph.txt
normalize com-dblp.ungraph.txt
normalize com-youtube.ungraph.txt
normalize com-lj.ungraph.txt
normalize com-orkut.ungraph.txt
# normalize com-friendster.ungraph.txt
