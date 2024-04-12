#!/usr/bin/bash
#SBATCH --job-name VOX
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu_exp
#SBATCH --time 01:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_voxpopuli_small_tokenizer.out

EXPERIMENT="ebranchformer_voxpopuli_cz_small_tokenizer"
PROJECT="voxpopuli"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/voxpopuli"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR || exit

python src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --preprocessing_num_workers="128" \
  --datasets_creation_config="${RECIPE_DIR}/voxpopuli_cz.json" \
  --writer_batch_size="200" \
  --tokenizer_name="Lakoc/voxpopuli_uni500_cz" \
  --vocab_size=500 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --pad_token="([pad])" \
  --unk_token="([unk])" \
  --bos_token="([bos])" \
  --eos_token="([eos])" \
  --mask_token="([mask])"
