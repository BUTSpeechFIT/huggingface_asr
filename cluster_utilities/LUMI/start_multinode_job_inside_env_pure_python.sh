#!/bin/bash -e
# Source bash profile
# shellcheck disable=SC1090
source ~/.bash_profile

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 1

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"
# Start conda environment inside the container
$WITH_CONDA


set -euo pipefail

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

if [ $SLURM_LOCALID -eq 0 ] ; then
  if command -v rocm-smi &> /dev/null ; then
    rm -rf /dev/shm/* || true
    rocm-smi || true	# rocm-smi returns exit code 2 even when it succeeds
  fi
else
  sleep 2
fi

# Run application
python "$@"
