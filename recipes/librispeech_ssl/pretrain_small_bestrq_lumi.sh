#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/librispeech_ssl/output_%x_%j.out"
#SBATCH --error="outputs/librispeech_ssl/output_%x_%j.err"
#SBATCH --partition=standard-g
#SBATCH --mem=0G
#SBATCH --time=2-00:00:00


EXPERIMENT="bestrq_lumi_libri_small_proper"
PROJECT="librispeech_ssl"

SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_ssl"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export OMP_NUM_THREADS=64
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
#export CXI_FORK_SAFE=1
#export CXI_FORK_SAFE_HP=1
#export FI_CXI_DISABLE_CQ_HUGETLB=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
#export FI_CXI_DEFAULT_CQ_SIZE=131072
#
#export ROCM_PATH=/opt/rocm
#export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

# Try playing with max_split_size_mb if you run into OOM errors.
#export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"


cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="128"
  --per_device_eval_batch_size="128"
  --dataloader_num_workers="8"
  --num_train_epochs="50"
  --group_by_length="True"
  --do_train
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-4"
  --warmup_steps="5000"
  --early_stopping_patience="5"
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
  --datasets_creation_config="${RECIPE_DIR}/librispeech_ssl.json"
  --writer_batch_size="50"
  --split_long_segments_to_chunks
  --cut_validation_from_train
  --validation_slice="10%"

  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing2d.json"

  # Model related arguments
--base_encoder_model="Lakoc/ebranchformer_6_128h_2d_bestrq_lessbooks"
--feature_extractor_name="Lakoc/fe_mel_wo_norm"

  )


srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env.sh" src/trainers/pretrain.py "${args[@]}"
