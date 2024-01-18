#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_preprocess.out

EXPERIMENT="ebranchformer_english_preprocess"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
ENV_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"

export HF_HOME="${ENV_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=128

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --preprocess_dataset_only

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.1"
  --length_column_name="input_len"
  --preprocessing_num_workers="64"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="1000"

)

python src/trainers/train_enc_dec_asr.py "${args[@]}"
