#!/bin/bash
#SBATCH --job-name=reval_slurp
#SBATCH --output=/pfs/lustrep3/users/yusufbol/logs/%x_%j.o
#SBATCH --error=/pfs/lustrep3/users/yusufbol/logs/%x_%j.e
#SBATCH --partition=standard-g
#SBATCH --account=project_465000970
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00
#SBATCH -D /pfs/lustrep3/users/yusufbol/code/huggingface_asr

N_GPUS=$SLURM_NTASKS_PER_NODE

EXPERIMENT=$SLURM_JOB_NAME
WORK_DIR="/pfs/lustrep3/users/yusufbol/code/huggingface_asr"
EXPS_DIR="/scratch/project_465000970/bolaji/"

RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
DATASETS="${RECIPE_DIR}/datasets_slurp_asr_lumi.json"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export WANDB_MODE=offline
export WANDB_PROJECT="eloquence_asr_llm"
export WANDB_ENTITY="butspeechfit"
export WANDB_RUN_ID="${EXPERIMENT}"

export HF_HOME=/scratch/project_465000970/bolaji/huggingface_cache/
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

ckpt=$1
if [ -z "$ckpt" ]; then
  echo "No checkpoint provided."
  exit 1
fi

# Get model name and number of steps from the checkpoint path
model_name=$(basename $(dirname $ckpt))
steps=$(basename $ckpt | cut -d'-' -f2)
EXPERIMENT_PATH="${EXPS_DIR}/revals/${model_name}_${steps}"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="8"   #"12" # 16
  --per_device_eval_batch_size="8" # 24
  --dataloader_num_workers="16"
  #--num_train_epochs="14"
  --max_steps="80000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
#  --do_train
  --do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="True"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-5"
  --warmup_steps="2000"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  #--lsm_factor="0.1"
  --gradient_accumulation_steps="2"

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
  --validation_split slurp_dev
  --test_splits slurp_dev slurp_test

  # Fisher context arguments
  --fisher_context_prefix='Transcribe the rest of the conversation given the following conversation history: "'
  --fisher_max_context=3
  #--fisher_context_trunc_to_shortest
  --prompt_prefix='" '
  --prompt_suffix=' Continued transcript: '

  # Preprocessing related arguments
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"

  # Model related arguments
  --from_pretrained="${ckpt}"

  #--feature_extractor_name="openai/whisper-small.en"
  #--base_encoder_model="openai/whisper-small.en"
  --feature_extractor_name="microsoft/wavlm-large"
  --base_encoder_model="microsoft/wavlm-large"
  --freeze_encoder="True"

  --tokenizer_name="allenai/OLMo-1B-hf"
  --base_decoder_model="allenai/OLMo-1B-hf"

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
  #--no_metrics
)

echo "Running training.."

srun \
  singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B $PWD:${RECIPE_DIR} \
    -B /scratch/project_465000970/ \
    $SIF ${RECIPE_DIR}/lumi_run_pytorch.sh src/trainers/alignment/train_ecd_lm_general_context.py "${args[@]}"