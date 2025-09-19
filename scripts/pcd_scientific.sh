#!/bin/bash

# Exit immediately if a command fails.
set -e

# Define your config values
SEEDS="1000 2000 3000 4000 5000"
TASKS="regex rfp zinc"  # molecule doesn't work on cluster
REWEIGHT_LOSS_OPTS="True False"
results_dir="/home/shrestj3/PCDiffusion/results"
model_dir="/scratch/work/shrestj3/off_moo_models"

# Create the full list of combinations
configs=()
for seed in $SEEDS; do
  for task in $TASKS; do
      for reweight in $REWEIGHT_LOSS_OPTS; do
        configs+=("$seed $task $reweight")
    done
  done
done

# Guard clause for SLURM
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

# Get the SLURM array task index
index=$SLURM_ARRAY_TASK_ID
config="${configs[$index]}"

if [ -z "$config" ]; then
    echo "Invalid config index: $index"
    exit 1
fi

seed=$(echo "$config" | awk '{print $1}')
task=$(echo "$config" | awk '{print $2}')
reweight_opt=$(echo "$config" | awk '{print $3}')

# Get GPU ID assigned by Slurm; default to 0 if not set
GPU_ID=${SLURM_LOCALID:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Handle the boolean flag conditionally
reweight_flag=""
if [[ "$reweight_opt" == "True" ]]; then
  reweight_flag="--reweight_loss"
fi

printf "\n--- Running Experiment --- \n"
printf "Seed: %s, Task: %s, Reweight: %s, GPU: %s\n" \
  "$seed" "$task" "$reweight_opt" "$GPU_ID"

# Run the Python Script with the parsed values
python train.py \
  --domain="scientific" \
  --seed="$seed" \
  --task_name="$task" \
  $reweight_flag \
  --save_dir="$results_dir" \
  --model_dir="$model_dir"

printf "\nAll experiments complete.\n"
