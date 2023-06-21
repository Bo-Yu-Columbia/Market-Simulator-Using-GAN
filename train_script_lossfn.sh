#!/bin/bash

# Define parameter ranges
loss_fn=(1 2 3 4 5 6 7 8)

# Loop over parameter combinations

for loss_fn in "${loss_fn[@]}"; do
    echo "Executing training command with loss_fn $loss_fn"
#    python train.py -use_cuda -total_steps 1 -datasets 'STOCKS' -algo 'SigCWGAN' 'GMMN' -loss_fn $loss_fn
    python evaluate.py -use_cuda -loss_fn $loss_fn
done

