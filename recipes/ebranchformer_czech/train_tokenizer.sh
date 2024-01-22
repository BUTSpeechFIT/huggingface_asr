#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_tokenizer.out

unset PYTHONPATH
unset PYTHONHOME
. /usr/local/share/Anaconda2/bin/activate python3
conda activate /mnt/matylda6/szoke/CONDA_ENVS/huggingface_asr


EXPERIMENT="ebranchformer_czech_small_v3" #wandb, dirname
PROJECT="tokenizer_czech_corpus" #wandb
WORK_DIR="/mnt/matylda6/szoke/HuggingFace/huggingface_asr/"
#ENV_DIR="/mnt/matylda6/szoke/CONDA_ENVS/huggingface_asr/"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_czech"

export HF_HOME="/mnt/scratch/tmp/szoke/huggingface_cache" # do not forget to set the HF token in this cache!
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"


#conda deactivate
#source activate loco_asr
#conda deactivate
#conda activate /mnt/matylda6/szoke/CONDA_ENVS/huggingface_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

python ./src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --preprocessing_num_workers="64" \
  --datasets_creation_config="${RECIPE_DIR}/datasets_tokenizer_fit.json" \
  --writer_batch_size="1000" \
  --tokenizer_name="szoke/czech_corpus_uni2000" \
  --vocab_size=2000 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --pad_token="<pad>" \
  --unk_token="<unk>" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --mask_token="<mask>"
