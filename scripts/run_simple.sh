#!/bin/bash

python main.py \
    --task center_locating \
    --method knn \
    --data_path /path/to/your/dataset \
    --data_type features \
    --lmda 10 \
    --K 1 \
    --weights_type inverse_distance \
    --lr 1e-5 \
    --max_iters 100 \
    --test