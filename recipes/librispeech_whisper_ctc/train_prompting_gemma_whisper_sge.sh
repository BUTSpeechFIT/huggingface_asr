#!/usr/bin/bash
#$ -N whisper_gemma
#$ -q long.q
#$ -l ram_free=180G,mem_free=180G
#$ -l scratch=8
#$ -l gpu=4,gpu_ram=40G
#$ -o /mnt/matylda5/ipoloka/projects/huggingface_asr/outputs/whisper_llm/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/matylda5/ipoloka/projects/huggingface_asr/outputs/whisper_llm/$JOB_NAME_$JOB_ID.err

PROJECT="whisper_llm_prompting"
EXPERIMENT="soft_prompts_gemma_whisper_medium_additional_linear_2b_it_v0.1"
SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
WORK_DIR=$SRC_DIR
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_whisper_ctc"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${PROJECT}/${EXPERIMENT}"

unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr

export PATH="/mnt/matylda5/ipoloka/utils/SCTK/bin:$PATH"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"

export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"

export WANDB_PROJECT=${PROJECT}
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"

# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="24"
  --per_device_eval_batch_size="24"
  --num_train_epochs="20"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --ddp_find_unused_parameters="False"

   # Data loader params
  --dataloader_num_workers="6"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="5e-5"
  --warmup_steps="1000"
  --early_stopping_patience="15"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="1"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --eval_steps="500"
  --save_steps="500"
  --wandb_predictions_to_save="500"
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_wer"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="1.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="8"
  --datasets_creation_config="${RECIPE_DIR}/librispeech_sge.json"
  --writer_batch_size="200"
  --test_splits librispeech_test.clean librispeech_test.other

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_pretrained="openai/whisper-medium"
  --tokenizer_name="openai/whisper-medium"
  --feature_extractor_name="openai/whisper-medium"
  --llm_model="google/gemma-2b-it"
  --asr_model_checkpoint="/mnt/matylda5/ipoloka/projects/huggingface_asr/experiments/whisper_llm/gemma-whisper-medium-learnable-blank-full-v2.1/checkpoint-40176/model.safetensors"
  --freeze_asr
  --freeze_llm
  --number_of_prompt_tokens=16


  # Generation related arguments
  --validation_slice="1920"
  --num_beams="1"
  --predict_with_generate="True"
)

"${SRC_DIR}/sge_tools/python" recipes/librispeech_whisper_ctc/whisper_llm_prompting.py "${args[@]}"
