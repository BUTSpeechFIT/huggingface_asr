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
export CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES


export NODENAME=$(cat /proc/sys/kernel/hostname)
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")

port_in_use() {
    lsof -i :$1 > /dev/null
}
MASTER_PORT=30000
STOP_CONDITION=40000
# Loop until the condition is met
while [ "$MASTER_PORT" -lt "$STOP_CONDITION" ]; do
    # Check if the port is in use
    if port_in_use "$MASTER_PORT"; then
        echo "Port $MASTER_PORT is already in use. Trying next port."
        MASTER_PORT=$((MASTER_PORT + 1))
    else
        # Port is not in use, export it and exit the loop
        export MASTER_PORT
        echo "Using MASTER_PORT: $MASTER_PORT"
        break
    fi
done

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Redirect stdout and stderr so that we get a prefix with the node name
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK err: /" >&2)


torchrun \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --node_rank="$SLURM_PROCID" \
  --master_addr="$MASTER_ADDR" \
   --master_port="$MASTER_PORT" \
   "$@"
