#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/librispeech_ssl/ssl_small_8gpus_bestrq_fixed.out

EXPERIMENT="ssl_small_8gpus_bestrq_fixed"
PROJECT="librispeech_ssl"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/librispeech_ssl"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="24"
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="24"
  --num_train_epochs="50"
  --group_by_length="True"
  --do_train
  --load_best_model_at_end
  --bf16

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="4e-3"
  --warmup_steps="5000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="5"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --greater_is_better="False"
  --save_total_limit="5"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/librispeech_ssl.json"
  --writer_batch_size="50"
  --split_long_segments_to_chunks
  --cut_validation_from_train
  --validation_slice="10%"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing2d_no_spec_aug.json"

  # Model related arguments
  --expect_2d_input
  --config_overrides="best_rq_num_books=8,best_rq_codebook_size=2048,best_rq_codebook_dim=16"
  --base_encoder_model="Lakoc/ebranchformer_12_256h_2d_bestrq"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k")

torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/pretrain_wav2vec2.py "${args[@]}"
