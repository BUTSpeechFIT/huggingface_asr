#!/usr/bin/bash
#SBATCH --job-name decred
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu_exp
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/decoding_ed_small_ood.out

EXPERIMENT="decoding_ed_small_ood"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

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
  --per_device_train_batch_size="256"
  --per_device_eval_batch_size="24"
  --dataloader_num_workers="4"
  --do_evaluate
  --learning_rate="5e-3"
  --logging_steps="1"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --report_to="wandb"
  --optim="adamw_torch"
  --dataloader_num_workers="24"
  --metric_for_best_model="eval_wer"
  --remove_unused_columns="False"
  --save_total_limit="1"
  --num_train_epochs="200"
  --greater_is_better="False"
  --group_by_length="False"
  --bf16
  --gradient_accumulation_steps="8"
  --early_stopping_patience="10"
  --load_best_model_at_end
  --lsm_factor="0.1"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
#  --dataset_name="/mnt/proj1/open-28-58/lakoc/huggingface_asr/test_orig_splits"
  --dataset_name="/mnt/proj1/open-28-58/lakoc/huggingface_asr/dataset_ood"
  --writer_batch_size="500"
  --test_splits fleurs_test gigaspeech_test ami_corpus_test
  --train_split="fleurs_test"
  --validation_split="ami_corpus_test"
  --text_transformations whisper_normalize_english
  --post_process_predicitons


  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
#  --from_pretrained="BUT-FIT/ED-small"
  --from_pretrained="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/mixing_weights_ood/checkpoint-1000"
#  --from_pretrained="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/checkpoint-234"
#  --from_pretrained="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/finetune_mixing_mechanism_linear_v2/checkpoint-58"
  --expect_2d_input
  --decoder_pos_emb_fixed

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
)

torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/train_enc_dec_asr.py "${args[@]}"
