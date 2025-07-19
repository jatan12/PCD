#!/bin/bash

# >>> Conda environment activation >>>
# eval "$(conda shell.bash hook)"
# conda activate off-moo
# <<< Conda environment activation <<<

# Synthetic Functions
synthetic="dtlz1 dtlz7 omnitest vlmop1 vlmop2 vlmop3 zdt1 zdt2 zdt3 zdt4 zdt6"

# RE
re="re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61"

# MORL
morl="mo_hopper_v2 mo_swimmer_v2"

# Scientific Design
scientific="molecule regex rfp zinc"

seeds="1000 2000 3000 4000 5000"
tasks="$synthetic $re $morl $scientific"
model="End2End"
train_modes="Vallina GradNorm PcGrad"

MAX_JOBS=2  #16
AVAILABLE_GPUS="0"  #"0 1"
MAX_RETRIES=0  #1
job_number=0

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

for seed in $seeds; do 
    for task in $tasks; do 
        for train_mode in $train_modes; do

        check_jobs
        gpu_allocation=$(get_gpu_allocation $job_number)
        ((job_number++))

        echo "Running seed=$seed task=$task train_mode=$train_mode on GPU $gpu_allocation"

        run_with_retry "offline_moo/off_moo_baselines/end2end/experiment.py \
            --model=${model} \
            --train_mode=${train_mode} \
            --task=${task} \
            --use_wandb=False \
            --retrain_model=False \
            --seed=${seed}" \
            "$gpu_allocation" & 

        done
    done 
done 

wait