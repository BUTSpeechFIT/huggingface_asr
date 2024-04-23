#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/voxpopuli_czech/output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --mem=60G
#SBATCH --time=1:00:00

EXPERIMENT="ebranchformer_cv_small_tokenizer"
SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/common_voice_cz"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"

cd $SRC_DIR || exit

srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH \
/runscripts/conda-python-simple src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --preprocessing_num_workers="8" \
  --datasets_creation_config="${RECIPE_DIR}/common_voice_cz.json" \
  --writer_batch_size="200" \
  --tokenizer_name="Lakoc/common_voice_uni50_cz" \
  --vocab_size=50 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --pad_token="([pad])" \
  --unk_token="([unk])" \
  --bos_token="([bos])" \
  --eos_token="([eos])" \
  --mask_token="([mask])"
