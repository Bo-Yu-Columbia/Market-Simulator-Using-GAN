#!/bin/bash

# Define parameter ranges
p_range=(2 3 4)
q_range=(2 3 4)
dims_range=("3*(50,)" "4*(50,)" "4*(50,)")

# Loop over parameter combinations
for p in "${p_range[@]}"; do
    for q in "${q_range[@]}"; do
        for dims in "${dims_range[@]}"; do
            echo "Executing training command with p=$p, q=$q, hidden_dims=$dims"
            python train.py -use_cuda -total_steps 1 -p $p -q $q -hidden_dims $dims -datasets 'STOCKS' -'SigCWGAN' 'GMMN'
        done
    done
done
