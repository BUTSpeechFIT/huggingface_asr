#!/bin/bash
#$ -N EU_ASR
#$ -q long.q@@blade
#$ -l ram_free=2G,mem_free=2G
#$ -l matylda5=0.1
#$ -pe smp 16
#$ -o /mnt/matylda5/ipoloka/projects/huggingface_asr/eu_asr.o
#$ -e /mnt/matylda5/ipoloka/projects/huggingface_asr/eu_asr.e

# Limit job runtime to 24 h -> 86400 s, send SIGXCPU and SIGKILL if limit is breached
ulimit -t 86400

# Enable opening multiple files
ulimit -n 8000

# Enable bigger arrow shards
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME

SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"

export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache/"
export PATH="/mnt/matylda5/ipoloka/utils:$PATH"

EXPERIMENT="cs_tokenizer"
PROJECT="cs_finetune"

RECIPE_DIR="${SRC_DIR}/recipes/cs_mix/sge"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"

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
  --datasets_creation_config="${RECIPE_DIR}/data_ft_normalized.json"
  --writer_batch_size="50"

  # Tokenizer related arguments
  --tokenizer_name="Lakoc/cz_ec_bpe1000"
  --vocab_size=997
  --tokenizer_type="BPE"
  --train_split="train"
  --pad_token="([pad])"
  --unk_token="([unk])"
  --bos_token="([bos])"
  --eos_token="([eos])"
  --mask_token="([mask])"
  --tokens_to_add "(LNG)" "(UNK)" "(SPN)"
)


N_GPUS="" ${SRC_DIR}/sge_tools/interactive_python src/trainers/train_tokenizer.py "${args[@]}"