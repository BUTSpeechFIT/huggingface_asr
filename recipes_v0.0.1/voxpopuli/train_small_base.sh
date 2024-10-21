#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_voxpopuli_small_base_150e.out

EXPERIMENT="ebranchformer_voxpopuli_small_base"
PROJECT="VoxPopuli"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/voxpopuli"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}_150e"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR || exit

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="64"
  --per_device_eval_batch_size="32"
  --dataloader_num_workers="24"
  --num_train_epochs="150"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --joint_decoding_during_training
  --load_best_model_at_end
  --restart_from="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/ebranchformer_voxpopuli_small_base/checkpoint-32650"
  --ignore_data_skip
  --metric_for_best_model="eval_wer"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="15000"
  --early_stopping_patience="50"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=500
  --greater_is_better="False"
  --save_total_limit="5"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="128"
  --datasets_creation_config="${RECIPE_DIR}/voxpopuli.json"
  --writer_batch_size="200"
  --test_splits voxpopuli_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/voxpopuli_uni500"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input

  # Generation related arguments
  --num_beams="4"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
  --eval_beam_factor="10"
)

torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/train_enc_dec_asr.py "${args[@]}"
