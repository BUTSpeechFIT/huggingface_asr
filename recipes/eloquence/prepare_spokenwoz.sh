#!/bin/bash
#$ -N spokenwoz_prep_collated_whisper2
#$ -q all.q@supergpu17
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda4=0.5,scratch=1,matylda6=0.5
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/spokenwoz_prep_collated_whisper2.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/spokenwoz_prep_collated_whisper2.e
EXPERIMENT="spokenwoz_prep_collated_whisper2"

# Job should finish in about 1 days
ulimit -t 100000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/isedlacek/projects/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/isedlacek/hugging-face"

python ${RECIPE_DIR}/prepare_spokenwoz.py
