#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --output="outputs/librispeech_aed/output_%x_%j.out"
#SBATCH --error="outputs/librispeech_aed/output_%x_%j.err"
#SBATCH --partition=small-g
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00

EXPERIMENT="baseline_ebranchformer_v2.3_fp32_continue"

SRC_DIR="/project/${EC_PROJECT}/ipoloka/huggingface_asr"
WORK_DIR="/scratch/${EC_PROJECT}/ipoloka/huggingface_asr"
RECIPE_DIR="${SRC_DIR}/recipes/librispeech_aed"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

module load LUMI partition/G PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209


# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

export HF_HOME="/flash/${EC_PROJECT}/ipoloka/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export WANDB_PROJECT="librispeech_aed"
export WANDB_RUN_ID="${EXPERIMENT}"
export WANDB_ENTITY="butspeechfit"


cd $SRC_DIR || exit

args=(
  # General training arguments
  --output_dir="${EXPERIMENT_PATH}"
  --per_device_train_batch_size="64"
  --per_device_eval_batch_size="32"
  --num_train_epochs="240"
  --group_by_length="True"
#  --bf16
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --ddp_find_unused_parameters="False"

   # Data loader params
  --dataloader_num_workers="6"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="40000"
  --early_stopping_patience="15"
  --weight_decay="1e-6"
  --max_grad_norm="1.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save="500"
  --greater_is_better="False"
  --save_total_limit="5"
  --metric_for_best_model="eval_librispeech_validation.other_wer"
  --push_to_hub_final_model
  --use_sclite_for_metrics

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="1.0"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="8"
  --datasets_creation_config="${RECIPE_DIR}/librispeech.json"
  --writer_batch_size="200"
  --test_splits librispeech_test.clean librispeech_test.other
  --pad_to_multiples_of="100"
  --merge_validation_splits="false"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/libri_5000_v2"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
#  --from_encoder_decoder_config
#  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
#  --base_decoder_model="Lakoc/gpt2_tiny_decoder_6_layers"
  --ctc_weight="0.3"
  --from_pretrained="/scratch/project_465000836/ipoloka/huggingface_asr/experiments/baseline_ebranchformer_v2.3_fp32/checkpoint-113300"

  # Generation related arguments
  --predict_with_generate
  --num_beams="1"
  --max_length="512"
  --decoding_ctc_weight="0"
  --eval_delay="2"
)


srun --unbuffered --kill-on-bad-exit singularity exec --bind /usr/sbin:/usr/sbin $SIFPYTORCH \
"${SRC_DIR}/cluster_utilities/LUMI/start_multinode_job_inside_env_pure_python.sh"  src/trainers/train_enc_dec_asr.py "${args[@]}"
