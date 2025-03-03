#!/bin/bash
#$ -N EU_ASR
#$ -q long.q@@blade
#$ -l ram_free=2G,mem_free=2G
#$ -l matylda5=0.1
#$ -pe smp 16
#$ -o /mnt/matylda5/ipoloka/projects/huggingface_asr/eu_asr.o
#$ -e /mnt/matylda5/ipoloka/projects/huggingface_asr/eu_asr.e

# Limit job runtime to 24 h -> 86400 s, send SIGXCPU and SIGKILL if limit is breached
ulimit -t 86400

# Enable opening multiple files
ulimit -n 8000

# Enable bigger arrow shards
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr

WORK_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
# Ensure work directory exists
cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit
}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache/"
export PATH="/mnt/matylda5/ipoloka/utils:$PATH"


METADATA_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr/metadata_dirs/eu_asr_cs/"
mkdir -p $METADATA_DIR
EUASR_TRAIN_CES1=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_EuasrElda_kaldi/data/Czech-Euasr-001_train
EUASR_TRAIN_CES2=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_EuasrElda_kaldi/data/Czech-Euasr-002
EUASR_DEV=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_EuasrElda_kaldi/data/Czech-Euasr-001_dev
EUASR_TEST=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_EuasrElda_kaldi/data/Czech-Euasr-001_test


ln -s $EUASR_TRAIN_CES1 $METADATA_DIR/train1
ln -s $EUASR_TRAIN_CES2 $METADATA_DIR/train2
ln -s $EUASR_DEV $METADATA_DIR/dev
ln -s $EUASR_TEST $METADATA_DIR/test

python src/dataset_builders/preprocess_dataset.py \
  --dataset_builder src/dataset_builders/kaldi_dataset \
  --metadata_dir $METADATA_DIR \
  --num_proc 16 \
  --splits train1 train2 dev test


METADATA_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr/metadata_dirs/cv_punct_cs/"
mkdir -p $METADATA_DIR
CV_TRAIN=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_CommonVoice_lhotse/kaldi_data/commonvoice_cs_train
CV_DEV=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_CommonVoice_lhotse/kaldi_data/commonvoice_cs_dev
CV_TEST=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_CommonVoice_lhotse/kaldi_data/commonvoice_cs_test

ln -s $CV_TRAIN $METADATA_DIR/train
ln -s $CV_DEV $METADATA_DIR/dev
ln -s $CV_TEST $METADATA_DIR/test


python src/dataset_builders/preprocess_dataset.py \
  --dataset_builder src/dataset_builders/kaldi_dataset \
  --metadata_dir $METADATA_DIR \
  --num_proc 16 \
  --splits train dev test


METADATA_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr/metadata_dirs/vox_punct_cs"
mkdir -p $METADATA_DIR
VOX_TRAIN=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_VoxPopuli_lhotse/kaldi_data/voxpopuli_cs_train
VOX_DEV=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_VoxPopuli_lhotse/kaldi_data/voxpopuli_cs_dev
VOX_TEST=/mnt/matylda5/iveselyk/EU-ASR_TENDER/EU-ASR-Codebase/data_preparation/CZECH_VoxPopuli_lhotse/kaldi_data/voxpopuli_cs_test

ln -s $VOX_TRAIN $METADATA_DIR/train
ln -s $VOX_DEV $METADATA_DIR/dev
ln -s $VOX_TEST $METADATA_DIR/test

python src/dataset_builders/preprocess_dataset.py \
  --dataset_builder src/dataset_builders/kaldi_dataset \
  --metadata_dir $METADATA_DIR \
  --num_proc 16 \
  --splits train dev test


