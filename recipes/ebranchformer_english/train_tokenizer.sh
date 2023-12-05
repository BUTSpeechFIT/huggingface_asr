#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_tokenizer.out

EXPERIMENT="ebranchformer_english_tokenizer"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
ENV_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"

export HF_HOME="${ENV_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

python src/trainers/train_tokenizer.py \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --output_dir=$EXPERIMENT_PATH \
  --length_column_name="input_len" \
  --preprocessing_num_workers="64" \
  --datasets_creation_config="${RECIPE_DIR}/datasets.json" \
  --writer_batch_size="1000" \
  --tokenizer_name="Lakoc/english_corpus_uni5000" \
  --vocab_size=5000 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --pad_token="<pad>" \
  --unk_token="<unk>" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --mask_token="<mask>"
