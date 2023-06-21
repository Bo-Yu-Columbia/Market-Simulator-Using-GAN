#!/bin/bash

# Parameters
use_cuda="-use_cuda"

# Loop to execute train command 10 times
for ((i=1; i<=10; i++))
do
    total_steps=$((i * 100))
    echo "Executing training command $i with total steps: $total_steps"
    python train.py $use_cuda -total_steps $total_steps
done
