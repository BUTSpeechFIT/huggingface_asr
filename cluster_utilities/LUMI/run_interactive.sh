#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=ju-standard-g
#SBATCH --mem=480G
#SBATCH --time=24:00:00


module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

singularity shell $SIF

$WITH_CONDA

EXPERIMENT="ebranchformer_voxpopuli_single_node_fast"
PROJECT="VoxPopuli"
SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/voxpopuli"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"


cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="96"
  --per_device_eval_batch_size="32"
  --dataloader_num_workers="7"
  --num_train_epochs="150"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --ignore_data_skip
  --metric_for_best_model="eval_wer"
#  --track_ctc_loss
#  --dataloader_persistent_workers
#  --restart_from="/scratch/project_465000836/ipoloka/huggingface_asr/experiments/ebranchformer_voxpopuli_small_lumi_single_node/checkpoint-5450"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="15000"
  --early_stopping_patience="50"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="none"
  --logging_steps="1"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=500
  --greater_is_better="False"
  --save_total_limit="5"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/voxpopuli.json"
  --writer_batch_size="200"
  --test_splits voxpopuli_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/voxpopuli_uni500"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
)



#module load LUMI PyTorch/2.1.0-rocm-5.6.1-python-3.10-singularity-20231123
export OMP_NUM_THREADS=64
export MPICH_GPU_SUPPORT_ENABLED=1
#export CXI_FORK_SAFE=1

export OMP_NUM_THREADS=64
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

#export NCCL_DEBUG=INFO
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

# Try playing with max_split_size_mb if you run into OOM errors.
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

#export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
#export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
#
## Set MIOpen cache to a temporary folder.
#if [ $SLURM_LOCALID -eq 0 ] ; then
#    rm -rf $MIOPEN_USER_DB_PATH
#    mkdir -p $MIOPEN_USER_DB_PATH
#fi

#
#export PATH="/flash/project_465000836/ipoloka/clean_python/bin:$PATH"
#torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/train_enc_dec_asr.py "${args[@]}"
#export PATH="/flash/project_465000836/ipoloka/conda_install_dir/bin/:$PATH"
#srun --gpus-per-task 2  \
# --cpus-per-task 14  \
# /project/project_465000836/ipoloka/huggingface_asr/run_jobs_lumi \
# src/trainers/train_enc_dec_asr.py "${args[@]}"

#export SINGULARITYENV_NCCL_DEBUG=1
#export SINGULARITYENV_NCCL_DEBUG_SUBSYS=
#export SINGULARITYENV_NCCL_ASYNC_ERROR_HANDLING=1
#srun --gpus-per-task $SLURM_GPUS_PER_TASK  \
# --cpus-per-task $SLURM_CPUS_PER_TASK  \
#  torchrun --standalone --nnodes=1 --nproc-per-node=8 src/trainers/train_enc_dec_asr.py "${args[@]}"
#srun --gpus-per-task 2  \
# --cpus-per-task 14  \
#singularity exec $SIFPYTORCH /project/project_465000836/ipoloka/huggingface_asr/run_jobs_lumi2 -u src/trainers/train_enc_dec_asr.py "${args[@]}"

srun \
  singularity exec $SIFPYTORCH \
    /project/project_465000836/ipoloka/huggingface_asr/conda_run -u src/trainers/train_enc_dec_asr.py "${args[@]}"


#!/bin/bash -e

# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 1

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Set MIOpen cache to a temporary folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 1

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"
# Start conda environment inside the container
$WITH_CONDA

# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# Set environment for the app
export MASTER_ADDR=$(/runscripts/get-master "$SLURM_NODELIST")
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID
#export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
#export CXI_FORK_SAFE=1

# Run application
python "$@"


#
#c=fe
#MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
#
#srun \
# --cpu-bind=mask_cpu:$MYMASKS \
#  singularity exec $SIFPYTORCH \
#    /project/project_465000836/ipoloka/huggingface_asr/conda_run -u src/trainers/train_enc_dec_asr.py "${args[@]}"
