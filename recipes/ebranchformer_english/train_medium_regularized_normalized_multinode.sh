#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --nodes 2
#SBATCH --time 1:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_medium_regularized_normalized.out

EXPERIMENT="multinode_test"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"
NCPU_PER_NODE=128
HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="32"
  --per_device_eval_batch_size="8"
  --dataloader_num_workers="24"
  --num_train_epochs="100"
  --group_by_length="True"
  --bf16
  --do_train
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
  --min_duration_in_seconds="0.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="500"
  --test_splits wsj_test fisher_swbd_test voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/ebranchformer_16l_512h"
  --base_decoder_model="Lakoc/gpt2_512h_8l_add_head6_04"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
)

export NPROC_PER_NODE=1
export PARENT=`/bin/hostname -s`
export MPORT=13000
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
export WORLD_SIZE=$SLURM_NTASKS

srun --cpus-per-task $NCPU_PER_NODE --gpus-per-task $NPROC_PER_NODE  \
/mnt/proj1/open-28-58/lakoc/huggingface_asr/recipes/multinode_training/start_single_node_job.sh \
$EXPERIMENT $PROJECT $WORK_DIR $RECIPE_DIR $HF_HOME "${args[@]}"
