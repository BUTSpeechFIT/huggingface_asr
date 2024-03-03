#!/usr/bin/bash
#$ -N ASR
#$ -q long.q@@gpu
#$ -l ram_free=32G,mem_free=32G
#$ -l scratch=2
#$ -l gpu=1,gpu_ram=40G
#$ -o /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/english_mixing_scalar_lr5_e2.o
#$ -e /mnt/matylda5/ipoloka/projects/LoCo-ASR/experiments/english_mixing_scalar_lr5_e2.e

# Job should finish in 24 hours
ulimit -t 86400

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr


SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
EXPERIMENT="english_mixing_scalar_lr5_e2"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"
DATASET_DIR="/mnt/scratch/tmp/ipoloka/full_dataset"

export WANDB_PROJECT="intermediate_mixing_v2"
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"

export PATH="/mnt/matylda5/ipoloka/utils/SCTK/bin:$PATH"
#export HF_DATASETS_OFFLINE=1
#export HF_HUB_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
export $(/mnt/matylda4/kesiraju/bin/gpus 1) || exit 1
ssh -N -D 9002 pcspeech4 &
SSH_PID=$!

export http_proxy=socks5://localhost:9002

export https_proxy=socks5://localhost:9002

cd $SRC_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="256"
  --auto_find_batch_size
  --per_device_eval_batch_size="32"
  --do_train
  --do_evaluate
  --learning_rate="5e-2"
  --logging_steps="1"
  --save_strategy="epoch"
  --early_stopping_patience="6"
  --evaluation_strategy="epoch"
  --report_to="wandb"
  --optim="adamw_torch"
  --dataloader_num_workers="6"
  --metric_for_best_model="eval_wer"
  --remove_unused_columns="False"
  --save_total_limit="2"
  --num_train_epochs="200"
  --greater_is_better="False"
  --group_by_length="False"
  --fp16
  --gradient_accumulation_steps="8"
  --load_best_model_at_end
  --restart_from="/mnt/matylda5/ipoloka/projects/huggingface_asr/experiments/english_mixing_scalar_lr5_e2/checkpoint-48"
  --max_steps="1"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="6"
  --dataset_name="${DATASET_DIR}"
  --writer_batch_size="50"
  --train_split="validation"
  --validation_split="validationXX"
  --validation_slice_seed="42"
  --cut_validation_from_train
  --validation_slice="30%"
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test


  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing2d_pure.json"

  # Model related arguments
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --from_pretrained="/mnt/matylda5/ipoloka/models/checkpoint-231248"
  --expect_2d_input
  --decoder_pos_emb_fixed
  --finetune_mixing_mechanism scalar

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0"
)

python src/trainers/train_enc_dec_asr.py "${args[@]}"

kill $SSH_PID