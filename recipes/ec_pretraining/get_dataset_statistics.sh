#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/ec_pretraining/output_%x_%j.out"
#SBATCH --error="outputs/ec_pretraining/output_%x_%j.err"
#SBATCH --partition=debug-g
#SBATCH --mem=100G
#SBATCH --time=0:30:00


module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

DATA_DIR="/flash/project_465000836/data"

# Iterate over each item in the current directory
for item in $DATA_DIR/*; do
  # Check if the item is a directory
  if [ -d "$item" ]; then
    echo "Processing directory: $item"
    # Change to the directory
    srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH /runscripts/conda-python-simple  /project/project_465000836/ipoloka/huggingface_asr/src/utilities/get_dataset_statistics.py  --dataset_path="${item}"  --length_column_name="input_len"
  fi
done
