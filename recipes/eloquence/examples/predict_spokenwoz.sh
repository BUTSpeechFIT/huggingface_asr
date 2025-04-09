#!/bin/bash
#$ -N predict_spokenwoz_multiwoz_lora_8_ua
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=4,gpu_ram=20G
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/predict_spokenwoz_multiwoz_lora_8_ua.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/predict_spokenwoz_multiwoz_lora_8_ua.e
N_GPUS=1 # TODO: for inference, use as many gpus as possible -- spokenwoz test takes about 30 minuts with four 24G gpus
EXPERIMENT="test_experiment" # TODO: Put your experiment name here

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment # TODO: Change this to your python environment
source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/isedlacek/projects/huggingface_asr" # TODO: Change this to your working directory
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
DATASETS="${RECIPE_DIR}/datasets_spokenwoz.json" # TODO: Change the path to the dataset in this .json file
export HF_HOME="/mnt/matylda6/isedlacek/hugging-face" # TODO: Change this to your huggingface home

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# TODO: If your cluster has internet access, remove the following lines
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="eloquence-dst"

# get the gpu # TODO: this gpu acquisition script is specific to our cluster, you should change it to your own
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_eval_batch_size="4" # 24
  --dataloader_num_workers="4"
  --group_by_length="True"
  --length_column_name="turn_index"
  --bf16
  --bf16_full_eval
  --do_generate
  --woz_use_agent_history

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  
  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="100.0"
  --min_duration_in_seconds="0.0"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  # TODO: The following arguments should be changed according to the dataset .json file you are using (in $DATASETS)
  --validation_split sa_multiwoz_dev
  --test_splits spokenwoz_test sa_multiwoz_dev
  --do_not_remove_columns audio wav_id turn_index text agent_text domains slots context 

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"
  # NOTE: if you want to use whisper, use the proper feature extractor config instead
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"

  # Model related arguments TODO: change these according to the model you want to use
  --feature_extractor_name="pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft" #microsoft/wavlm-large
  --tokenizer_name="pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft" #allenai/OLMo-1B-hf
  --from_pretrained="pirxus/wavlm-large_olmo1b_lora_r16a16_np_ua_swft"

  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=400
)

echo "Running with args: ${args[@]}"

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm_woz_new.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm_woz_new.py "${args[@]}"
fi
