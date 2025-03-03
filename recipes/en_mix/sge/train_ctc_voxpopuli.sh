#!/bin/bash
#$ -N EC_CTC_EN
#$ -cwd
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -l gpu=4,gpu_ram=20G
#$ -o /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.err

#set -eux
# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=4

PROJECT="en_finetune"
SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"

RECIPE_DIR="${SRC_DIR}/recipes/en_mix/sge"

EXPERIMENT="200M_ec_ebranchformer__ctc_voxpopuli_v2"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"

export OMP_NUM_THREADS=16
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="32"
  --per_device_eval_batch_size="32"
  --dataloader_num_workers=4
  --dataloader_pin_memory=True
  --dataloader_prefetch_factor=2
  --num_train_epochs="200"
  --group_by_length="True"
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --bf16
  --auto_find_batch_size

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="2000"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --gradient_accumulation_steps="2"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --eval_delay=1
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_wer_mapped"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/datasets_normalized_vox_train.json"
  --writer_batch_size="50"
  --test_splits wsj_test fisher_swbd_test voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_processing.json"

  # Model related arguments
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --from_pretrained="/mnt/matylda6/szoke/EU-ASR/EU-ASR-Codebase/model_pretraining/models/EBF-BRQ-194M-nonSTR-L24-H512-CB4x1024x16-80x512x2d-M06_MVN_MCV-VP-LELD_LR3e4_wu30k/checkpoint-368000_ep3/"
  )

"${SRC_DIR}/sge_tools/interactive_python" "${SRC_DIR}/src/trainers/train_ctc_asr.py" "${args[@]}"
