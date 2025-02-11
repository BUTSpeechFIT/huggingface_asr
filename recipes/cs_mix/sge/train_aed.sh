#!/bin/bash
#$ -N EC_CTC_CZ
#$ -cwd
#$ -v SRC_ROOT
#$ -v WORK_DIR
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -l gpu=4,gpu_ram=20G
#$ -o /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/$USER/EC/exp/log/$JOB_NAME_$JOB_ID.err

set -eux
# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=1

# e.g. submit_sge.sh "+decode=mt_asr/mt_nsf ++training.per_device_eval_batch_size=1"
echo $SRC_ROOT
[ -z "$SRC_ROOT" ] && { echo "Please export SRC_ROOT"; exit 1; }



EXPERIMENT="90M_ebranchformer_ctc"
EXPERIMENT_PATH="${WORK_DIR}/${EXPERIMENT}"

PROJECT="czech_ctc"

export OMP_NUM_THREADS=16
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"

cd $SRC_ROOT || exit


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="12"
  --per_device_eval_batch_size="24"
  --dataloader_num_workers="4"
  --num_train_epochs="50"
  --group_by_length="True"
  --do_train
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="3e-4"
  --warmup_steps="2000"
  --early_stopping_patience="3"
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

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --pad_to_multiples_of="100"
  --datasets_creation_config="${SRC_ROOT}/recipes/cs_mix/data.json"
  --writer_batch_size="50"

  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_ROOT}/recipes/cs_mix/data_processing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/bpe1000_cz"
  --feature_extractor_name="/mnt/matylda6/szoke/EU-ASR/MODELS/k-cs2_dap_mvn_brq_ebf_90M-L18-H384-CB4x1024x16-80x384x2d-M06_LR5e4_wu20k_2n_intmed/checkpoint-50000_ep1"
  --from_pretrained="/mnt/matylda6/szoke/EU-ASR/MODELS/k-cs2_dap_mvn_brq_ebf_90M-L18-H384-CB4x1024x16-80x384x2d-M06_LR5e4_wu20k_2n_intmed/checkpoint-50000_ep1"
  )

"${SRC_ROOT}/sge_tools/interactive_python" "${SRC_ROOT}/src/trainers/train_ctc_asr.py" "${args[@]}"
