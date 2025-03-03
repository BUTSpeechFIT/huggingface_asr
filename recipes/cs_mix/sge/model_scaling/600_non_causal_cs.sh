#!/bin/bash
#$ -N EC_CTC_CZ
#$ -cwd
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -l gpu=3,gpu_ram=40G
#$ -o /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.err

set -eux
# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=3

PROJECT="cs_finetune_scaling"
SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"

RECIPE_DIR="${SRC_DIR}/recipes/cs_mix/sge"

EXPERIMENT="600M___non_causal"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"

export OMP_NUM_THREADS=16
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="10"
  --per_device_eval_batch_size="10"
  --dataloader_num_workers=2
  --dataloader_pin_memory=True
  --dataloader_prefetch_factor=1
  --num_train_epochs="200"
  --group_by_length="True"
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --bf16
  --restart_from="/mnt/matylda5/ipoloka/projects/huggingface_asr/experiments/600M___non_causal/checkpoint-17108"
  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="8e-5"
  --warmup_steps="0"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --gradient_accumulation_steps="2"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_wer_mapped"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.5"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --pad_to_multiples_of="100"
  --datasets_creation_config="${RECIPE_DIR}/data_ft_normalized.json"
  --writer_batch_size="50"
  --test_splits euasr_cs_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_processing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/cz_ec_bpe1000"
  --feature_extractor_name="/mnt/matylda5/ipoloka/projects/huggingface_asr/ec_cz_fe"
  --from_pretrained="/mnt/matylda6/szoke/EU-ASR/EU-ASR-Codebase/model_pretraining/models/EBF-BRQ-641M-nonSTR-L36-H768-CB4x1024x16-80x768x2d-M06_MVN_MCV-VP-LELD_LR1e4_wu40k/checkpoint-228000_ep3/"
  )

"${SRC_DIR}/sge_tools/interactive_python" "${SRC_DIR}/src/trainers/train_ctc_asr.py" "${args[@]}"
