#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output="outputs/voxpopuli_czech/output_%x_%j.txt"
#SBATCH --partition=standard-g
#SBATCH --mem=0G
#SBATCH --time=24:00:00
#SBATCH --exclusive

EXPERIMENT="voxpopuli_czech3_pretrain_small_original_causal"
SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/voxpopuli/czech"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

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
export OMP_NUM_THREADS=8

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64


export HF_HOME="/scratch/${EC_PROJECT}/ipoloka/hf_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT="voxpopuli_czech3"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit


srun --unbuffered --kill-on-bad-exit singularity exec --bind /usr/sbin:/usr/sbin $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env.sh"  src/trainers/run_pretraining.py \
	--dataset_name="facebook/voxpopuli" \
	--dataset_config_names cs \
	--dataset_split_names train \
	--model_name_or_path="Lakoc/ebranchformer_6_128h_for_pretraining_2d" \
	--output_dir="./wav2vec2-pretrained-demo_ebranchformer" \
	--max_train_steps="20000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="1" \
	--learning_rate="0.005" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="20.0" \
	--min_duration_in_seconds="2.0" \
	--logging_steps="1" \
	--saving_steps="1000" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--mask_time_prob="0.65" \
	--mask_time_length="10" \
	--pad_to_multiple_of=100
