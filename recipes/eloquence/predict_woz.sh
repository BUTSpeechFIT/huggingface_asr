#!/bin/bash
#$ -N predict_spokenwoz_multiwoz_nolora
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=4,gpu_ram=16G
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/predict_spokenwoz_multiwoz_nolora.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/eloquence/dst/predict_spokenwoz_multiwoz_nolora.e
N_GPUS=4
EXPERIMENT="predict_spokenwoz_multiwoz_nolora"

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/isedlacek/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
#DATASETS="${RECIPE_DIR}/datasets_woz.json"
DATASETS="${RECIPE_DIR}/datasets_woz_eval.json"
#DATASETS="${RECIPE_DIR}/datasets_multiwoz.json"
#DATASETS="${RECIPE_DIR}/datasets_spokenwoz.json"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/isedlacek/hugging-face"

export WANDB_DISABLED=true
export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="eloquence-asr"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="12" # 20
  --per_device_eval_batch_size="6" # 24
  --dataloader_num_workers="4"
  --group_by_length="True"
  --length_column_name="turn_index"
  --bf16
  --bf16_full_eval
  --do_generate
  --qformer_eval_callback
  --ddp_find_unused_parameters="False"

  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="100.0"
  --min_duration_in_seconds="0.0"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --validation_split dev
  --test_splits spokenwoz_test spokenwoz_dev sa_multiwoz_test
  --do_not_remove_columns audio wav_id turn_index text agent_text domains slots context 

  --slurp_dump_pred
  
  # Preprocessing related arguments
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"

  # Model related arguments
  #--from_pretrained=""
  #--restart_from="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_libri_how2/checkpoint-16000/"
  #--from_pretrained="/mnt/matylda5/iyusuf/exps/eloquence/ehpc_62_dump/bolaji/exp/wll_olmo1b_general_context_asr_fisher_libri_how2_train_enc_context0_labels_nostr/checkpoint-38000"
  #--from_pretrained="/mnt/scratch/tmp/isedlacek/models/phase1_ft_nc" # base connector

  --from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/spokenwoz_multiwoz/checkpoint-16000"

  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/spokenwoz_multiwoz_lora_test/checkpoint-8000"
  #--decoder_lora
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/multiwoz_only/checkpoint-4000"

  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/spokenwoz_sanity/checkpoint-12000"
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/spokenwoz_ft_single/checkpoint-12000"
  #--restart_from="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/spokenwoz_ft_original_tr/checkpoint-6000"
  # latest qlogin /mnt/matylda6/isedlacek/projects/huggingface_asr/exp/test/checkpoint-8000

  #--feature_extractor_name="openai/whisper-small.en"
  #--base_encoder_model="openai/whisper-small.en"
  --feature_extractor_name="microsoft/wavlm-large"
  --base_encoder_model="microsoft/wavlm-large"
  --freeze_encoder="True"

  --tokenizer_name="allenai/OLMo-1B-hf"
  --base_decoder_model="allenai/OLMo-1B-hf"
  
  --connector_type='encoder_stacked'
  --downsampling_factor=6
  --conn_hidden_size=1024
  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=4096

  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=200
  --predict_with_generate
  #--no_metrics
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm_spokenwoz.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm_spokenwoz.py "${args[@]}"
fi
