#!/bin/bash
#$ -N EC_RNNT_CZ
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

PROJECT="cs_finetune_v2"
SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"

RECIPE_DIR="${SRC_DIR}/recipes/cs_mix/sge"

EXPERIMENT="90M_ebranchformer_rnnt2e4_causal"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"

export OMP_NUM_THREADS=16
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="8"
  --per_device_eval_batch_size="1"
  --dataloader_num_workers=4
  --dataloader_pin_memory=False
  --dataloader_prefetch_factor=1
  --num_train_epochs="200"
  --group_by_length="True"
  --do_train
  --do_evaluate
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-4"
  --warmup_steps="2000"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_wer_mapped"
  --eval_delay=2

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.5"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/data_ft_normalized_extended.json"
  --writer_batch_size="50"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_processing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/cz_ec_bpe1000"
  --feature_extractor_name="/mnt/matylda5/ipoloka/projects/huggingface_asr/ec_cz_fe"
  --from_pretrained="/mnt/matylda6/szoke/EU-ASR/MODELS/k-fd3_dap_mvn_brq_ebf_90M-L18-H384-CB4x1024x16-80x384x2d-M06_LR2e4_wu40k_intmed/checkpoint-855000_ep7"
  
 )

"${SRC_DIR}/sge_tools/interactive_python" "${SRC_DIR}/src/trainers/train_rnnt_asr.py" "${args[@]}"
