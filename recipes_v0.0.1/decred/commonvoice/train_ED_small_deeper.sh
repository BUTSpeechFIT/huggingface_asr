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

EXPERIMENT="ED_small_cv_en_deeper"
SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/decred/commonvoice"
EXPERIMENT_PATH="${WORK_DIR}/experiments/decred/commonvoice/${EXPERIMENT}"

module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209

export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export OMP_NUM_THREADS=16

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

export HF_HOME="/scratch/${EC_PROJECT}/ipoloka/hf_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT="decred_commonvoice_en"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="256"
  --per_device_eval_batch_size="256"
  --num_train_epochs="50"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --eval_delay="5"
  --push_to_hub_final_model

   # Data loader params
  --dataloader_num_workers="6"
  --dataloader_pin_memory="True"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="10000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=50
  --greater_is_better="False"
  --save_total_limit="2"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --datasets_creation_config="${RECIPE_DIR}/common_voice_en.json"
  --writer_batch_size="200"
  --test_splits common_voice_13_en_common_voice_13_en_test
  --pad_to_multiples_of="100"
  --load_pure_dataset_only

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/common_voice_uni1000"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --config_overrides="decoder_n_layer=8;encoder_num_hidden_layers=16"

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
  --override_for_evaluation="ctc_weight=0.3;num_beams=10"
)

srun --unbuffered --kill-on-bad-exit singularity exec --bind /usr/sbin:/usr/sbin $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env_pure_python.sh"  src/trainers/train_enc_dec_asr.py "${args[@]}"
