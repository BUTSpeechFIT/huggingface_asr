#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output="outputs/librispeech/ssl/output_%x_%j.out"
#SBATCH --error="outputs/librispeech/ssl/output_%x_%j.err"
#SBATCH --partition=standard
#SBATCH --mem=200G
#SBATCH --time=02:00:00


EXPERIMENT="cs_tokenizer"
PROJECT="cs_finetune"

SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/cs_mix"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export OMP_NUM_THREADS=64

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"


cd $SRC_DIR || exit


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --pad_to_multiples_of="100"
  --datasets_creation_config="${RECIPE_DIR}/data_tokenizer.json"
  --writer_batch_size="50"

  # Tokenizer related arguments
  --tokenizer_name="Lakoc/bpe1000_cz_ec" \
  --vocab_size=1000 \
  --tokenizer_type="BPE" \
  --train_split="train" \
  --pad_token="([pad])" \
  --unk_token="([unk])" \
  --bos_token="([bos])" \
  --eos_token="([eos])" \
  --mask_token="([mask])"
  )


srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH \
"/runscripts/conda-python-simple" src/trainers/train_tokenizer.py "${args[@]}"