#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/voxpopuli_czech/output_%x_%j.txt"
#SBATCH --partition=ju-standard-g
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

EXPERIMENT="ebranchformer_voxpopuli_czech"
PROJECT="VoxPopuli"
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

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64


export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"


cd $SRC_DIR || exit



python src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --preprocessing_num_workers="128" \
  --datasets_creation_config="${RECIPE_DIR}/voxpopuli_cz.json" \
  --writer_batch_size="200" \
  --tokenizer_name="Lakoc/voxpopuli_uni500_cz" \
  --vocab_size=500 \
  --tokenizer_type="unigram" \
  --text_column_name="text" \
  --train_split="train" \
  --pad_token="([pad])" \
  --unk_token="([unk])" \
  --bos_token="([bos])" \
  --eos_token="([eos])" \
  --mask_token="([mask])"

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="96"
  --per_device_eval_batch_size="256"
  --num_train_epochs="50"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --load_best_model_at_end

   # Data loader params
  --dataloader_num_workers="6"
  --dataloader_persistent_workers="True"
  --dataloader_pin_memory="True"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="15000"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --eval_delay="10"
  --wandb_predictions_to_save="500"
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_wer"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --datasets_creation_config="${RECIPE_DIR}/voxpopuli_cz.json"
  --writer_batch_size="200"
  --test_splits voxpopuli_test
  --pad_to_multiples_of="100"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/voxpopuli_uni500_cz"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input


  # Generation related arguments
  --num_beams="4"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
  --override_for_evaluation="ctc_weight=0.3;num_beams=10"
)

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"


srun --unbuffered --kill-on-bad-exit  singularity exec $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env_pure_python.sh" src/trainers/train_enc_dec_asr.py "${args[@]}"
