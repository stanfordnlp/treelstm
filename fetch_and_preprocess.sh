#!/bin/bash
python2.7 scripts/download.py
python2.7 scripts/preprocess-sick.py
python2.7 scripts/preprocess-sst.py

glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    th scripts/convert-wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
        $glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi
