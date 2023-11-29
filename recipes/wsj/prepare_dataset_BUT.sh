#!/bin/bash
#$ -N Fisher_prep
#$ -q long.q@@blade
#$ -l ram_free=0.2G,mem_free=0.2G
#$ -l matylda5=0.05,matylda4=0.05
#$ -pe smp 16
#$ -o /mnt/matylda5/ipoloka/projects/huggingface_asr/wsj_v2.o
#$ -e /mnt/matylda5/ipoloka/projects/huggingface_asr/wsj_v2.e

# Limit job runtime to 24 h -> 86400 s, send SIGXCPU and SIGKILL if limit is breached
ulimit -t 86400

# Enable opening multiple files
ulimit -n 4000

# Enable bigger arrow shards
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr

# Ensure work directory exists
METADATA_DIR="/mnt/matylda4/kesiraju/tools/kaldi-asr/kaldi/egs/wsj/s5/data/"
WORK_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit
}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${WORK_DIR}/hf_cache"

python src/dataset_builders/preprocess_dataset.py \
  --dataset_builder src/dataset_builders/kaldi_dataset \
  --metadata_dir $METADATA_DIR \
  --num_proc 16 \
  --splits train_si284 dev_dt_20 dev_dt_05 test_eval92
