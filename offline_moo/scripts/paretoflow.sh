#!/bin/bash

# Define your config values
synthetic="dtlz1 dtlz7 omnitest vlmop1 vlmop2 vlmop3 zdt1 zdt2 zdt3 zdt4 zdt6"
re="re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61"
morl="mo_hopper_v2 mo_swimmer_v2"  # Not used in the cluster
scientific="molecule regex rfp zinc"  # molecule doesn't work on cluster

tasks="rfp"  # maybe rfp as well
seeds="4000 5000"

# Create the full list of combinations
configs=()
for seed in $seeds; do 
    for task in $tasks; do 
        configs+=("$seed $task")
    done
done

# Set GPU ID (default 0)
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Iterate over each configuration and run the experiment
for config in "${configs[@]}"; do
    seed=$(echo $config | awk '{print $1}')
    task=$(echo $config | awk '{print $2}')
    echo "Running seed=$seed task=$task on GPU $GPU_ID"
    python offline_moo/off_moo_baselines/paretoflow/experiment.py \
        --task="${task}" \
        --fm_epochs=1000 \
        --fm_batch_size=128 \
        --seed="${seed}"
done
