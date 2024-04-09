#!/bin/bash -e

# Source bash profile
# shellcheck disable=SC1090
source ~/.bash_profile

# Make sure GPUs are up
if [ "$SLURM_LOCALID" -eq 0 ] ; then
    rocm-smi
fi

# Start conda environment inside the container
$WITH_CONDA

# Set environment for the app
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")
export MASTER_PORT=29501
export CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

torchrun \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --node_rank="$SLURM_PROCID" \
  --master_addr="$MASTER_ADDR" \
   --master_port="$MASTER_PORT" \
   "$@"
