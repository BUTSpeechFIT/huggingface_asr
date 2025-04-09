#!/bin/bash
#$ -N spokenwoz_multiwoz_lora_r16a16_bs96_np_ua
#$ -q all.q@supergpu*
#$ -l h=!supergpu14
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=2,gpu_ram=20G
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/spokenwoz_multiwoz_lora_r16a16_bs96_np_ua.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/spokenwoz_multiwoz_lora_r16a16_bs96_np_ua.e
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
  --per_device_train_batch_size="4" # 20
  --per_device_eval_batch_size="4" # 24
  --dataloader_num_workers="4"
  --max_steps="30000"
  --group_by_length="True"
  --length_column_name="turn_index"
  --bf16
  --bf16_full_eval
  --do_train
  --do_generate

  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="False"
  --decoder_lora # NOTE: will add lora adapters to the decoder after loading the asr checkpoint
  --decoder_lora_rank 16
  --decoder_lora_alpha 16
  --prompt_prefix="" # since we're adding lora, no need for instruction really..
  --woz_use_agent_history

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="5e-5"
  --warmup_steps="500"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --gradient_accumulation_steps="8"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --save_steps="1000"
  --eval_steps="1000"
  --wandb_predictions_to_save=100 # 60
  --greater_is_better="False"
  --metric_for_best_model="eval_loss"
  --save_total_limit="3"

  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="100.0"
  --min_duration_in_seconds="0.0"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --validation_split spokenwoz_dev
  --test_splits spokenwoz_dev spokenwoz_test
  --do_not_remove_columns audio wav_id turn_index text agent_text domains slots context 

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"

  # Model related arguments
  --from_pretrained="/mnt/scratch/tmp/isedlacek/models/phase1_ft_nc" # base connector ASR model (wavlm + olmo1b FT NC)
  --from_pretrained="pirxus/wavlm-large_olmo1b_phase1_ft_nc" # base connector ASR model (wavlm + olmo1b FT NC)
  --tokenizer_name="allenai/OLMo-1B-hf"
  --feature_extractor_name="microsoft/wavlm-large"
  --freeze_encoder="True"

  --connector_type='encoder_stacked'
  --downsampling_factor=6
  --conn_hidden_size=1024
  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=4096
  
  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=400
  --predict_with_generate
  --no_metrics
)

echo "Running with args: ${args[@]}"

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm_woz_new.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm_woz_new.py "${args[@]}"
fi
