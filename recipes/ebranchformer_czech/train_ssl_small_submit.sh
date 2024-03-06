#!/usr/bin/bash
#SBATCH --job-name SSL
#SBATCH --account OPEN-28-57
#SBATCH --gpus 8
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-57/szoke/huggingface_asr/recipes/ebranchformer_czech/ssl-train.%A.out  # %A is the job ID, %a is the array task ID
#SBATCH --error=/mnt/proj1/open-28-57/szoke/huggingface_asr/recipes/ebranchformer_czech/ssl-train.%A.err


ml Anaconda3/2023.07-2

#unset PYTHONPATH
#unset PYTHONHOME
#. /usr/local/share/Anaconda2/bin/activate python3
#conda activate /mnt/matylda6/szoke/CONDA_ENVS/huggingface_asr

WORK_DIR="/mnt/proj1/open-28-57/szoke/huggingface_asr"
ENV_DIR="/mnt/proj1/open-28-57/szoke/CONDA_ENVS/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_czech"
EXPERIMENT="ebranchformer_czech_small_v1" #wandb, dirname
PROJECT="czech_ssl" #wandb

export HF_HOME="/scratch/project/open-28-57/szoke/huggingface_cache" # do not forget to set the HF token in this cache!
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"





[ -z "${CONDA_EXE}" ] && echo "Error, missing $CONDA_EXE !" && exit 1
CONDA_BASE=$(${CONDA_EXE} info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda deactivate
#source /apps/all/Anaconda3/2023.07-2/bin/activate /mnt/proj1/open-28-57/szoke/CONDA_ENVS/huggingface_asr
conda activate /mnt/proj1/open-28-57/szoke/CONDA_ENVS/huggingface_asr

#conda deactivate
#conda activate /mnt/matylda6/szoke/CONDA_ENVS/huggingface_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR





args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="4"
  --per_device_eval_batch_size="4"
  --dataloader_num_workers="24"
  --num_train_epochs="50"
  --group_by_length="True"
  --bf16
  --do_train
  --load_best_model_at_end
  --auto_find_batch_size=True

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-5"
  --warmup_steps="2000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="1"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --greater_is_better="False"
  --save_total_limit="5"

  # Data related arguments
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="2.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="1"
  --datasets_creation_config="${RECIPE_DIR}/datasets_ssl_split/datasets_ssl.json"
  --writer_batch_size="100"
  --validation_split=train
  --validation_slice=100

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --base_encoder_model="iszoke/ebranchformer_12_256h"
  --feature_extractor_name="facebook/wav2vec2-xls-r-1b"
)

torchrun --standalone --nnodes=1 --nproc-per-node=${SLURM_GPUS_ON_NODE} /mnt/proj1/open-28-57/szoke/huggingface_asr/src/trainers/pretrain_wav2vec2.py "${args[@]}"
#python src/trainers/pretrain_wav2vec2.py "${args[@]}"
