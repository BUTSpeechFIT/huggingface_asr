#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output="outputs/librispeech_ssl/output_%x_%j.out"
#SBATCH --error="outputs/librispeech_ssl/output_%x_%j.err"
#SBATCH --partition=standard
#SBATCH --mem=200G
#SBATCH --time=02:00:00


EXPERIMENT="libri_stats"
PROJECT="librispeech_ssl"

SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_ssl"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export OMP_NUM_THREADS=64
export MPICH_GPU_SUPPORT_ENABLED=1

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"


cd $SRC_DIR || exit
mkdir -p "$EXPERIMENT_PATH"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="64"


  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --pad_to_multiples_of="100"
  --datasets_creation_config="${RECIPE_DIR}/librispeech_ssl.json"
  --writer_batch_size="50"
  --split_long_segments_to_chunks

  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing2d_pure.json"
  --feature_extractor_name="Lakoc/fe_mel_80_global_stats"

  )


srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH \
"/runscripts/conda-python-simple" src/trainers/compute_dataset_statistics.py "${args[@]}"
