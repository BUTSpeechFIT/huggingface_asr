#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time 10:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/preprocess_english.out

EXPERIMENT="preprocess_english"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"
HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"


export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}_restart_proper"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="500"
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"
)

python src/trainers/train_enc_dec_asr.py "${args[@]}" --preprocess_dataset_only
