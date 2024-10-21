#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/decred/output_%x_%j.out"
#SBATCH --error="outputs/decred/output_%x_%j.err"
#SBATCH --partition=small-g
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00

EXPERIMENT="WhisperOOD"
SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/decred/out_of_domain"
EXPERIMENT_PATH="${WORK_DIR}/experiments/decred/out_of_domain/${EXPERIMENT}"

module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export OMP_NUM_THREADS=16

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

export HF_HOME="/scratch/${EC_PROJECT}/ipoloka/hf_out"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"


cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="16"
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="24"
  --num_train_epochs="100"
  --group_by_length="True"
  --do_evaluate

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="500"
  --test_splits fleurs_test gigaspeech_test ami_corpus_test
  --train_split="train"
  --validation_split="ami_corpus_test"
  --text_transformations whisper_normalize_english

  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing_whisper.json"

  # Model related arguments
  --tokenizer_name="openai/whisper-medium"
  --feature_extractor_name="openai/whisper-medium"
  --from_pretrained="openai/whisper-medium"

  # Generation related arguments
  --post_process_predicitons
  --num_beams="1"
  --max_length="448"
  --predict_with_generate
)


srun --unbuffered --kill-on-bad-exit singularity exec --bind /usr/sbin:/usr/sbin $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env_pure_python.sh"  src/trainers/train_enc_dec_asr.py "${args[@]}"
