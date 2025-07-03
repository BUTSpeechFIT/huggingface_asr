#!/bin/bash
#$ -N general_context_asr_training
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=2,gpu_ram=20G
#$ -l h=!(supergpu5|supergpu8|supergpu7|supergpu15)
#$ -o /mnt/matylda3/isvecjan/workspace/speechlm.hf_asr/exp/wavlm-llama32_1b-pretrain.o
#$ -e /mnt/matylda3/isvecjan/workspace/speechlm.hf_asr/exp/wavlm-llama32_1b-pretrain.e
echo "Hostname: ${HOSTNAME}" >&2
N_GPUS=2
EXPERIMENT="wavlm_olmo1b"

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
# source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr
source /mnt/matylda3/isvecjan/miniconda3/bin/activate /mnt/matylda3/isvecjan/miniconda3/envs/speechlm-0425

# WORK_DIR="/mnt/matylda6/isedlacek/projects/huggingface_asr"
WORK_DIR=/mnt/matylda3/isvecjan/workspace/speechlm.hf_asr
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
#DATASETS="${RECIPE_DIR}/datasets.json"
#DATASETS="${RECIPE_DIR}/datasets_libri_how2.json"
#DATASETS="${RECIPE_DIR}/datasets_lc.json"
#DATASETS="${RECIPE_DIR}/datasets_how2.json"
#DATASETS="${RECIPE_DIR}/datasets_fisher_ctx.json"
DATASETS="${RECIPE_DIR}/datasets_context.but.json"
# DATASETS="${RECIPE_DIR}/datasets_context_but.fisher_how2.json" #${RECIPE_DIR}/datasets_context_but_mlang.fleurs.json

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export WANDB_MODE=offline
export WANDB_PROJECT="eloquence_asr_llm"
export WANDB_RUN_ID="${EXPERIMENT}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export HF_HOME="/mnt/matylda6/isedlacek/hugging-face"
export HF_HOME=/mnt/scratch/tmp/isvecjan/hf_home


# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="8"   #"12" # 16
  --per_device_eval_batch_size="8" # 24
  --dataloader_num_workers="4"
  #--num_train_epochs="14"
  --max_steps="80000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="True"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="5e-5"
  --warmup_steps="2000"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  #--lsm_factor="0.1"
  --gradient_accumulation_steps="4"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --save_steps="2000"
  --eval_steps="2000"
  --wandb_predictions_to_save=100 # 60
  --greater_is_better="False"
  --metric_for_best_model="eval_loss"
  --save_total_limit="3"

  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="8"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --validation_split val
  --test_splits val fisher_test librispeech_test how2_dev5

  # Fisher context arguments
  --fisher_context_prefix='Transcribe the rest of the conversation given the following conversation history: "'
  --fisher_max_context=0
  #--fisher_context_trunc_to_shortest
  --prompt_prefix='" '
  --prompt_suffix=' Continued transcript: '

  # Preprocessing related arguments
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"

  # Model related arguments
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_fisher_wavlm/checkpoint-26000"
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_libri_how2_cont/checkpoint-42000"
  #--restart_from="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_libri_how2/checkpoint-16000/"
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wlml_stte_olmo1b_context_turns_fixed/checkpoint-40000"

  #--feature_extractor_name="openai/whisper-small.en"
  #--base_encoder_model="openai/whisper-small.en"
  --feature_extractor_name="microsoft/wavlm-large"
  --base_encoder_model="microsoft/wavlm-large"
  --freeze_encoder="False"

  --tokenizer_name="allenai/OLMo-1B-hf"
  --base_decoder_model="allenai/OLMo-1B-hf"

  --connector_type='encoder_stacked'
  --downsampling_factor=6
  --conn_hidden_size=1024
  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=4096

  #--connector_type='linear_stacked'
  #--downsampling_factor=5
  #--conn_hidden_size=2048
  #--qf_intermediate_size=4096

  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=170
  --predict_with_generate
  #--no_metrics
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm_general_context.py "${args[@]}"
else
  python -u src/trainers/alignment/train_ecd_lm_general_context.py "${args[@]}"
fi
