#!/usr/bin/bash
EXPERIMENT=$1
PROJECT=$2
WORK_DIR=$3
RECIPE_DIR=$4
HF_HOME=$5
args=("${@:6}")

export HF_HOME=$HF_HOME
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_TASK \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_PROCID \
  --master_addr="$PARENT" \
   --master_port="$MPORT" \
   src/trainers/train_enc_dec_asr.py "${args[@]}"
