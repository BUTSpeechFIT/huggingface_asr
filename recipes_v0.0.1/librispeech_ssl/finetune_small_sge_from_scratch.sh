#!/usr/bin/bash
#$ -N LS_EC
#$ -q long.q
#$ -l ram_free=90G,mem_free=90G
#$ -l scratch=4
#$ -l gpu=2,gpu_ram=40G
#$ -o /mnt/scratch/tmp/ipoloka/outputs/ssl_ec_libri/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/scratch/tmp/ipoloka/outputs/ssl_ec_libri/$JOB_NAME_$JOB_ID.err

EXPERIMENT="libri_small_from_scratch_v3"
PROJECT="librispeech_ssl_ft"

SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
WORK_DIR="/mnt/scratch/tmp/ipoloka/hf_exp"
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_ssl"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"



unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr/

export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"

export HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache/"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"
export N_GPUS=2

cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="32"
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="4"
  --num_train_epochs="50"
  --group_by_length="True"
  --do_train
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="2000"
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
  --datasets_creation_config="${RECIPE_DIR}/librispeech.json"
  --writer_batch_size="50"

  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing2d.json"

  # Model related arguments
  --tokenizer_name="Lakoc/libri_1000"
  --feature_extractor_name="Lakoc/fe_mel_wo_norm"
  --base_encoder_model="Lakoc/ebranchformer_6_128h_2d_bestrq_lessbooks"

#  --from_pretrained="/mnt/scratch/tmp/ipoloka/hf_exp/experiments/bestrq_lumi_libri_small_proper_v5/checkpoint-76456"
#  --freeze_encoder
  --config_overrides="apply_spec_augment=False"
  )


PORT=9049

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Port forwarding is already running."
else
    # Start port forwarding
    ssh -N -D $PORT pcspeech4 &
    echo "Port forwarding started on port $PORT."
fi

# If N_GPUS is set, export devices
if [ -n "$N_GPUS" ]; then
  export $(/mnt/matylda4/kesiraju/bin/gpus $N_GPUS) || exit 1
  echo "Visible devices: ${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES=""
fi


export http_proxy=socks5://localhost:$PORT
export https_proxy=socks5://localhost:$PORT
export PATH="/mnt/matylda5/ipoloka/utils/SCTK/bin:$PATH"
# if more than one device is passed, use torchrun to run the script
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/train_ctc_asr.py "${args[@]}"
else
  python src/trainers/train_ctc_asr.py "${args[@]}"
fi
