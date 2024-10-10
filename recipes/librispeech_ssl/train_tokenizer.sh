#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_tokenizer.out


PORT=9049

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Port forwarding is already running."
else
    # Start port forwarding
    ssh -N -D $PORT pcspeech4 &
    echo "Port forwarding started on port $PORT."
fi
export http_proxy=socks5://localhost:$PORT
export https_proxy=socks5://localhost:$PORT

unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr/
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache/"

SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_ssl"

python src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="2.0" \
  --length_column_name="input_len" \
  --remove_unused_columns="False" \
  --preprocessing_num_workers="16" \
  --pad_to_multiples_of="100" \
  --datasets_creation_config="${RECIPE_DIR}/librispeech.json" \
  --writer_batch_size="50" \
  --tokenizer_name="Lakoc/libri100" \
  --vocab_size=100 \
  --tokenizer_type="BPE" \
  --train_split="train" \
  --pad_token="([pad])" \
  --unk_token="([unk])" \
  --bos_token="([bos])" \
  --eos_token="([eos])" \
  --mask_token="([mask])"
