#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time 10:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/english_model_decode.out

EXPERIMENT="english_model_decode"
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
  --per_device_train_batch_size="32"
  --per_device_eval_batch_size="2"
  --dataloader_num_workers="24"
  --num_train_epochs="100"
  --group_by_length="True"
  --bf16
  --do_evaluate
  --joint_decoding_during_training
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-3"
  --warmup_steps="40000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="0.5"
  --lsm_factor="0.1"
  --mask_unks
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="none"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=500
  --greater_is_better="False"
  --save_total_limit="5"
  --track_ctc_loss

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/datasets_decoding_test.json"
  --writer_batch_size="500"
  --test_splits voxpopuli_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --from_pretrained="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/ebranchformer_english_medium_regularized_normalized/checkpoint-378950"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input

  # Generation related arguments
  --num_beams="5"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
)

python src/trainers/train_enc_dec_asr.py "${args[@]}"
