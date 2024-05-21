#!/bin/bash -e
# Source bash profile
# shellcheck disable=SC1090
# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"
# Start conda environment inside the container
$WITH_CONDA

set -euo pipefail

export NODENAME=$(cat /proc/sys/kernel/hostname)
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")

export MASTER_PORT=$(comm -23 <(seq 49152 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)

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
