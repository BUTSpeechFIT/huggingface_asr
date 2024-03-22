#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu_exp
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/decode_whisper_small_greedy.out

EXPERIMENT="decode_whisper_small_greedy"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"
HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"


export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_eval_batch_size="8"
  --dataloader_num_workers="24"
  --do_evaluate

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="128"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="1000"
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test
  --text_transformations whisper_normalize_english clean_special_tokens_english_train transforms_unfinished_words_to_unks_train


  # Preprocessing related arguments
  --data_preprocessing_config="${WORK_DIR}/configs/default_data_preprocessing_whisper.json"

  # Model related arguments
  --tokenizer_name="openai/whisper-small"
  --feature_extractor_name="openai/whisper-small"
  --from_pretrained="openai/whisper-small"

  # Generation related arguments
  --post_process_predicitons
  --filter_empty_labels
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
)

torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/train_enc_dec_asr.py "${args[@]}"
