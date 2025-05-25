#!/bin/bash
# This script is used to train the model with the specified parameters.

# Encoder
# WavLM Large: wavlm
# Whisper L3: wl3
# Whisper L3 turbo: wl3t
# Wav2vec2-XLSR: xlsr
# Seamless M4T: slm4t
# mHuBERT: mhubert
enc=wl3t

# Decoder
# OLMo2 7b Instruct: olmo2-7b-it
# Gemma2 9b Instruct: gemma2-9b-it
# Gemma3 1b Instruct: gemma3-1b-it
# Gemma3 4b Instruct: gemma3-4b-it
# Gemma3 12b Instruct: gemma3-12b-it
# Salamandra2 2b Instruct: salamandra2-2b-it
# Salamandra2 7b Instruct: salamandra2-7b-it
# LLama3.2 3b Instruct: llama3-3b-it
dec=s2b
# dec=s2bit

gpus=4
batch=4
grad_acc=1
lr=5e-5
dataset=fleurs5
# dataset=fleurs-speechmassive
validation_split=val
test_splits="val fisher_test librispeech_test how2_dev5"
# validation_split=validation
# test_splits="fleurs_test fleurs_test_el_gr fleurs_test_en_us fleurs_test_es_419 fleurs_test_it_it fleurs_test_sr_rs"
prompt_suffix="Continued transcript"
# prompt_suffix="Continued transcript in"
info=

work_dir="/mnt/matylda3/isvecjan/workspace/speechlm.hf_asr"

# Parse cmd line options
. parse_options.sh || exit 1

recipe_dir="${work_dir}/recipes/eloquence"
experiment=${enc}_${dec}_${dataset}_b$((${gpus} * ${batch} * ${grad_acc}))_lr${lr}${info}
experiment_path="${work_dir}/exp"

if [[ $batch -eq 1 ]]; then
  gpu_ram=16G
elif [[ $batch -eq 2 ]]; then
  gpu_ram=20G
elif [[ $batch -eq 4 ]]; then
  gpu_ram=30G
elif [[ $batch -eq 8 ]]; then
  gpu_ram=40G
# elif [[ $batch -eq 16 ]]; then
#   gpu_ram=40G
else
  echo "Unknown batch size: $batch"
  exit 1
fi

# Dataset
if [[ $dataset == "english" ]]; then
  datasets="\${RECIPE_DIR}/datasets_context_but.fisher_how2_librispeech.json"
elif [[ $dataset == "fleurs5" ]]; then
  datasets="\${RECIPE_DIR}/datasets_context_but.fleurs5.json"
  validation_split=validation
  test_splits="validation fleurs_test fleurs_test_el_gr fleurs_test_en_us fleurs_test_es_419 fleurs_test_it_it fleurs_test_sr_rs"
elif [[ $dataset == "fleurs13" ]]; then
  datasets="\${RECIPE_DIR}/datasets_context_but.fleurs13.json"
  validation_split=validation
  test_splits="fleurs_test fleurs_test_en_us fleurs_test_es_419 fleurs_test_fr_fr \
  fleurs_test_ar_eg fleurs_test_de_de fleurs_test_hu_hu fleurs_test_ko_kr fleurs_test_nl_nl \
  fleurs_test_pl_pl fleurs_test_pt_br fleurs_test_ru_ru fleurs_test_tr_tr fleurs_test_vi_vn"
elif [[ dataset == "cv17-5" ]]; then
  datasets="\${RECIPE_DIR}/datasets_context_but.cv17-5.json"
elif [[ dataset == "cv21-5" ]]; then
  datasets="\${RECIPE_DIR}/datasets_context_but.cv21-5.json"
else
  echo "Unknown dataset: $dataset"
  exit 1
fi

if [[ $enc == "wl3" ]] || [[ $enc == "wl3t" ]]; then
  data_preprocessing_config="\${RECIPE_DIR}/data_preprocessing_whisper.json"
else
  echo "Unknown data processor: $data_processor"
  exit 1
fi

# Encoder
if [[ $enc == "wl3" ]]; then
  encoder="openai/whisper-large-v3"
elif [[ $enc == "wl3t" ]]; then
  encoder="openai/whisper-large-v3-turbo"
else
  echo "Unknown encoder: $encoder"
  exit 1
fi

# Decoder
if [[ $dec == "s2b" ]]; then
  decoder="BSC-LT/salamandra-2b"
elif [[ $dec == "s2bit" ]]; then
  decoder="BSC-LT/salamandra-2b-instruct"
else
  echo "Unknown decoder: $decoder"
  exit 1
fi

echo """#!/bin/bash
#$ -N $experiment
#$ -q long.q
#$ -l ram_free=40G,mem_free=40G
#$ -l scratch=1
#$ -l h=!(supergpu5|supergpu8|supergpu7)
#$ -l gpu=$gpus,gpu_ram=$gpu_ram
#$ -o $experiment_path/$experiment.o
#$ -e $experiment_path/$experiment.e
######
#SBATCH --job-name=mhubert_gemma3-4b-it_mlang3-pretrain
#SBATCH --output=/gpfs/projects/ehpc148/honzas/speechlm.hf_asr/logs/%x_%j.out
#SBATCH --error=/gpfs/projects/ehpc148/honzas/speechlm.hf_asr/logs/%x_%j.err
#SBATCH --account=ehpc148
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/gpfs/projects/ehpc148/honzas/speechlm.hf_asr
echo "Hostname: \${HOSTNAME}" >&2
N_GPUS=$gpus
EXPERIMENT=$experiment

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# module add anaconda/2024.02
# source /apps/ACC/ANACONDA/2024.02/bin/activate /gpfs/projects/ehpc148/honzas/miniconda3/speechlm-0425
source /mnt/matylda3/isvecjan/miniconda3/bin/activate /mnt/matylda3/isvecjan/miniconda3/envs/speechlm-0425

WORK_DIR=$work_dir  # /mnt/matylda3/isvecjan/workspace/speechlm.hf_asr
# WORK_DIR=/gpfs/projects/ehpc148/honzas/speechlm.hf_asr #/home/vut/vut860362/proj/speechlm.hf_asr

EXPERIMENT_PATH=$experiment_path #"\${WORK_DIR}/exp/\${EXPERIMENT}"
RECIPE_DIR="\${WORK_DIR}/recipes/eloquence"
# DATASETS="\${RECIPE_DIR}/datasets.json"
# DATASETS="\${RECIPE_DIR}/datasets_libri_how2.json"
# DATASETS="\${RECIPE_DIR}/datasets_lc.json"
# DATASETS="\${RECIPE_DIR}/datasets_how2.json"
# DATASETS="\${RECIPE_DIR}/datasets_fisher_ctx.json"
# DATASETS="\${RECIPE_DIR}/datasets_context_bsc.json"
# DATASETS="\${RECIPE_DIR}/datasets_context_bsc_mlang1.json"
# DATASETS="\${RECIPE_DIR}/datasets_context_bsc_mlang2.json"

DATASETS=$datasets #"\${RECIPE_DIR}/datasets_context_but_mlang.fleurs.json"
# DATASETS="\${RECIPE_DIR}/datasets_context_but_mlang.cv21.json"

# DATASETS="\${RECIPE_DIR}/datasets_context_but_mlang3_cv.json"
# DATASETS="\${RECIPE_DIR}/datasets_context_but_mlang3_fleurs.json"
# DATASETS=\${RECIPE_DIR}/datasets_context.but.json

cd \$WORK_DIR || {
  echo "No such directory \$WORK_DIR"
  exit 1
}

export PYTHONPATH="\${PYTHONPATH}:\${WORK_DIR}/src"
export WANDB_MODE=offline
export WANDB_PROJECT="eloquence_asr_llm"

export WANDB_RUN_ID="\${EXPERIMENT}"

# export HF_HOME=/scratch/project_465001737/bolaji/huggingface_cache/
# export HF_HOME=/gpfs/projects/ehpc148/honzas/hf_home
export HF_HOME=/mnt/scratch/tmp/isvecjan/hf_home
#export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=\$(free-gpus.sh \$N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=\$EXPERIMENT_PATH/\$EXPERIMENT
  --per_device_train_batch_size="$batch"
  --per_device_eval_batch_size="$batch"
  --dataloader_num_workers=2
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
  --learning_rate="$lr"
  --warmup_steps="2000"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  # --lsm_factor="0.1"
  --gradient_accumulation_steps=$grad_acc

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
  --datasets_creation_config="\${DATASETS}"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="8"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --validation_split $validation_split
  --test_splits $test_splits

  # Fisher context arguments
  --fisher_context_prefix='Transcribe the rest of the conversation given the following conversation history: \"'
  --fisher_max_context=0
  #--fisher_context_trunc_to_shortest
  --prompt_prefix='\" '
  --prompt_suffix=' $prompt_suffix '



  # Preprocessing related arguments
  --data_preprocessing_config=$data_preprocessing_config
  # "\${RECIPE_DIR}/data_preprocessing_whisper.json"
  # --data_preprocessing_config="\${RECIPE_DIR}/data_preprocessing_wavlm.json"
  # --data_preprocessing_config="\${RECIPE_DIR}/data_preprocessing_seamless.json"


  # Model related arguments
#--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_fisher_wavlm/checkpoint-26000"
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_libri_how2_cont/checkpoint-42000"
  #--restart_from="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wsm_olmo1b_stte_w2000_libri_how2/checkpoint-16000/"
  #--from_pretrained="/mnt/matylda6/isedlacek/projects/huggingface_asr/exp/wlml_stte_olmo1b_context_turns_fixed/checkpoint-40000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo1b_general_context_asr_fisher_libri_how2_train_enc_context0_labels_nostr/checkpoint-38000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo1b_general_context_asr_fisher_libri_how2_freeze_enc_context0_labels_nostr/checkpoint-28000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo7b_instruct_general_context_asr_fisher_libri_how2_freeze_enc_context0_labels_nostr/checkpoint-44000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo7b_instruct_general_context_asr_fisher_libri_how2_train_enc_context0_labels_nostr/checkpoint-24000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo1b_general_context_asr_fisher_libri_how2_freeze_enc_context0_labels_nostr_lr2e5/checkpoint-56000"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo1b_general_context_asr_fisher_libri_how2_train_enc_context0_labels_nostr_lr2e5/checkpoint-40000/"
#  --from_pretrained="/gpfs/projects/ehpc62/bolaji//exp/wll_olmo7b_instruct_general_context_asr_fisher_libri_how2_freeze_enc_context0_labels_nostr_lr2e5/checkpoint-24000"
  # --feature_extractor_name="openai/whisper-small"
  # --base_encoder_model="openai/whisper-small"
  --feature_extractor_name=$encoder #"openai/whisper-large-v3"
  --base_encoder_model=$encoder # "openai/whisper-large-v3"

  #--restart_from="\$EXPERIMENT_PATH/checkpoint-\$\(cat \$EXPERIMENT_PATH/latest\)"

  # ENCODER
  # --feature_extractor_name="microsoft/wavlm-large"
  # --base_encoder_model="microsoft/wavlm-large"

  # --feature_extractor_name=facebook/wav2vec2-large-xlsr-53
  # --base_encoder_model=facebook/wav2vec2-large-xlsr-53

  # --feature_extractor_name=facebook/seamless-m4t-v2-large
  # --base_encoder_model=facebook/seamless-m4t-v2-large

  # --feature_extractor_name=utter-project/mHuBERT-147
  # --base_encoder_model=utter-project/mHuBERT-147
  --freeze_encoder="False"


  # DECODER
  # --tokenizer_name="allenai/OLMo-2-1124-7B-Instruct"
  # --base_decoder_model="allenai/OLMo-2-1124-7B-Instruct"

  # --tokenizer_name="google/gemma-2-9b-it"
  # --base_decoder_model="google/gemma-2-9b-it"

  # --tokenizer_name=google/gemma-3-1b-it
  # --base_decoder_model=google/gemma-3-1b-it
  # --tokenizer_name="google/gemma-3-4b-it"
  # --base_decoder_model="google/gemma-3-4b-it"
  # --tokenizer_name="google/gemma-3-12b-it"
  # --base_decoder_model="google/gemma-3-12b-it"

  --tokenizer_name=$decoder #BSC-LT/salamandra-2b-instruct
  --base_decoder_model=$decoder # BSC-LT/salamandra-2b-instruct
  # --tokenizer_name="BSC-LT/salamandra-7b-instruct"
  # --base_decoder_model="BSC-LT/salamandra-7b-instruct"

  # --tokenizer_name=meta-llama/Llama-3.2-3B-Instruct
  # --base_decoder_model=meta-llama/Llama-3.2-3B-Instruct
  ##########


  # --tokenizer_name="google/gemma-7b-it"
  # --base_decoder_model="google/gemma-7b-it"


  #--tokenizer_name="allenai/OLMo-2-1124-7B-Instruct"
  #--base_decoder_model="google/gemma-7b-it"

  #--tokenizer_name="allenai/OLMo-2-1124-7B-Instruct"
  #--base_decoder_model="allenai/OLMo-2-1124-7B-Instruct"

  #--tokenizer_name="allenai/OLMo-2-1124-13B-Instruct"
  #--base_decoder_model="allenai/OLMo-2-1124-13B-Instruct"

  # --tokenizer_name="google/gemma-7b-it"
  # --base_decoder_model="google/gemma-7b-it"


  # --tokenizer_name="allenai/OLMo-1B-hf"
  # --base_decoder_model="allenai/OLMo-1B-hf"

  #  --tokenizer_name="allenai/OLMo-7B-Instruct-hf"
  #  --base_decoder_model="allenai/OLMo-7B-Instruct-hf"

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
  # --no_metrics
)

echo "Running training ..."
if [ "\$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=\$N_GPUS src/trainers/alignment/train_ecd_lm_general_context.py \"\${args[@]}\"
else
  python -u src/trainers/alignment/train_ecd_lm_general_context.py \"\${args[@]}\"
fi
""" > $work_dir/exp/$experiment.sh

qsub $work_dir/exp/$experiment.sh
