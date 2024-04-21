#!/bin/bash

python ../main.py \
    --task center_locating \
    --method dummy_classifier \
    --data_path ../data/ms1 \
    --data_type features \
    --lmda 10 \
    --K 1 \
    --weights_type inverse_distance \
    --lr 1e-5 \
    --max_iters 100 \
    --test \
    --Kmin 1 \
    --Kmax 1 \
    --lmdaMin 10 \
    --lmdaMax 10 \
    --lrmin 1e-2 \
    --lrmax 1e-2 \
    --n_folds 1 \
    --n_params 1
