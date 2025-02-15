import multiprocessing
from dataclasses import dataclass, field
from typing import List, Optional, Union

import multiprocess
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    _argument_group_name = "Model related arguments"
    """Initialization related arguments."""
    base_encoder_model: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    base_decoder_model: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    from_pretrained: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    init_encoder: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models to initialize the encoder with"}
    )
    from_config: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model/config or model/config identifier from huggingface.co/models"}
    )
    from_encoder_decoder_config: Optional[bool] = field(
        default=False, metadata={"help": "Whether to create model from encoder and decoder configs."}
    )
    replace_aligned_decoder: Optional[bool] = field(
        default=False, metadata={"help": "Whether to replace the decoder of the model specified in from_pretrained."}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10;resid_pdrop=0.2;scale_attn_weights=false;summary_type=cls_index"
            )
        },
    )
    average_checkpoints: Optional[bool] = field(default=False, metadata={"help": "Whether to average checkpoints."})

    """Model architecture related arguments."""
    expect_2d_input: Optional[bool] = field(default=False, metadata={"help": "Whether to expect 2d input for encoder."})
    ctc_weight: Optional[float] = field(default=0, metadata={"help": "Weight of CTC loss."})
    lsm_factor: Optional[float] = field(default=0, metadata={"help": "Label smoothing coefficient for CE loss."})
    shared_lm_head: Optional[bool] = field(default=False, metadata={"help": "Whether to share LM head params."})
    decoder_pos_emb_fixed: Optional[bool] = field(default=False, metadata={"help": "Whether to disable decoder WPE."})

    """Whisper specific arguments."""
    whisper_language: Optional[str] = field(default=None, metadata={"help": "Language of the model."})
    whisper_task: Optional[str] = field(default=None, metadata={"help": "Task of the model."})

    """Token mixing specific arguments."""
    finetune_mixing_mechanism: Optional[str] = field(
        default=None, metadata={"help": "Type of mixing mechanism to use for finetuning."}
    )

    """WavLM-specific arguments"""
    layer_to_extract: Optional[int] = field(
        default=None,
        metadata={"help": "Intermediate layer from which to extract encoder features; defaults to output layer"}
    )
    freeze_encoder: Optional[bool] = field(default=True, metadata={"help": "Whether to freeze the encoder."})


@dataclass
class GeneralTrainingArguments(Seq2SeqTrainingArguments):
    _argument_group_name = "Training related arguments"
    """Arguments related to phases of the training."""
    preprocess_dataset_only: bool = field(default=False, metadata={"help": "Whether to preprocess dataset only"})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})
    do_evaluate: Optional[bool] = field(default=False, metadata={"help": "Whether to run evaluation."})
    do_generate: Optional[bool] = field(default=False, metadata={"help": "Whether to run generation."})
    do_generate_sequential: Optional[bool] = field(default=False, metadata={"help": "Whether to run generation for DST in sequential fashion."})
    restart_from: Optional[str] = field(
        default="", metadata={"help": "Path to checkpoint used to restart the training."}
    )

    """Preprocessing and postprocessing related arguments."""
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_source_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path for the encoder source language."}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    collator_rename_features: Optional[bool] = field(
        default=True,
        metadata={"help": "Rename input_features to input_values in collator."},
    )
    """Arguments changing behavior of the training."""
    early_stopping_patience: Optional[int] = field(default=-1, metadata={"help": "Patience for early stopping."})
    track_ctc_loss: Optional[bool] = field(default=False, metadata={"help": "Whether to log CTC loss."})
    joint_decoding_during_training: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use joint decoding during training."}
    )
    freeze_encoder_epochs: Optional[int] = field(default=None, metadata={"help": "Freezes the encoder for the specified number of epochs."})
    qformer_eval_callback: Optional[bool] = field(default=False, metadata={"help": "Makes sure the encoder and decoder for a qformer model are put into eval mode for training."})
    qf_enc_unfreeze_epochs: Optional[int] = field(default=None, metadata={"help": "Unfreezes the qformer model encoder after a certain number of epochs."})
    qf_dec_unfreeze_epochs: Optional[int] = field(default=None, metadata={"help": "Unfreezes the qformer model decoder after a certain number of epochs."})
    qf_pretrain_epochs: Optional[int] = field(default=0, metadata={"help": "Number of pretraining epochs for the Qformer."})
    mask_unks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to mask unknown tokens for cross entropy."}
    )
    use_start_method_spawn: Optional[bool] = field(
        default=False, metadata={"help": "Whether multiprocessing should be started by spawn"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_start_method_spawn:
            multiprocessing.set_start_method("spawn", force=True)
            # pylint: disable=no-member
            multiprocess.set_start_method("spawn", force=True)
            self.dataloader_persistent_workers = True
            super().__post_init__()


@dataclass
class PretrainingArguments(GeneralTrainingArguments):
    _argument_group_name = "Pretraining related arguments"
    gumbel_temperature_decay: Optional[float] = field(default=0.999995, metadata={"help": "Gumbel temperature decay."})
    min_gumbel_temperature: Optional[float] = field(default=0.5, metadata={"help": "Minimum Gumbel temperature."})
    max_gumbel_temperature: Optional[float] = field(default=2.0, metadata={"help": "Maximum Gumbel temperature."})


@dataclass
class GenerationArguments:
    _argument_group_name = "Generation related arguments"
    """General generation arguments."""
    num_beams: Optional[int] = field(default=1, metadata={"help": "Num beams for decoding."})
    max_length: Optional[int] = field(default=200, metadata={"help": "Max number of generated tokens."})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max number of tokens generated after the prompt for decoder-only models."})
    length_penalty: Optional[float] = field(default=1.0, metadata={"help": "Length penalty for generation."})
    early_stopping: Optional[Union[str, bool]] = field(
        default=False, metadata={"help": "Whether to use early stopping."}
    )
    eval_beam_factor: Optional[int] = field(
        default=1, metadata={"help": "Factor to increase beam size for evaluation."}
    )
    """Joint decoding related arguments."""
    decoding_ctc_weight: Optional[float] = field(default=0.0, metadata={"help": "CTC weight to bias hypothesis."})
    ctc_margin: Optional[float] = field(default=0, metadata={"help": "Margin to stop generation."})
    lm_model: Optional[str] = field(default=None, metadata={"help": "Path to external LM."})
    lm_weight: Optional[float] = field(default=0.0, metadata={"help": "Weight of external LM."})
    """Generation logging related arguments."""
    wandb_predictions_to_save: Optional[int] = field(
        default=100, metadata={"help": "Number of predictions to save to wandb."}
    )
    num_predictions_to_return: Optional[int] = field(default=1, metadata={"help": "Number of predictions to return."})
    nbest_path_to_save: Optional[str] = field(default="nbests", metadata={"help": "Path to save nbest hypotheses."})
    save_output_states: Optional[bool] = field(default=False, metadata={"help": "Whether to save output states."})
    low_memory: Optional[bool] = field(default=False, metadata={"help": "Whether to use sequential beam search."})
    post_process_predictions: Optional[bool] = field(
        default=False, metadata={"help": "Whether to post process predictions."}
    )
    apply_eos_space_trick: Optional[bool] = field(default=False, metadata={"help": "Whether to apply eos space trick."})
    eos_space_trick_weight: Optional[float] = field(default=0.0, metadata={"help": "Weight of eos space trick."})
    space_token_id: Optional[int] = field(default=-1, metadata={"help": "Space token id."})
    override_for_evaluation: Optional[str] = field(
        default=None,
        metadata={"help": "Arguments to override for evaluation. Example: " "decoding_ctc_weight=0.3;lm_model=gpt2"},
    )
    no_metrics: Optional[bool] = field(default=False, metadata={"help": "Disables generation and metric computation in the evaluate loop. Useful for speeding up the evaluation."})


@dataclass
class DataTrainingArguments:
    _argument_group_name = "Data related arguments"
    """Dataset source related arguments."""
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "The config of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to the data directory, used with specific dataset builders."}
    )
    datasets_creation_config: Optional[str] = field(
        default=None, metadata={"help": "Path to dictionary containing setups for multiple datasets."}
    )
    """Dataset preprocessing related arguments."""
    data_preprocessing_config: Optional[str] = field(
        default=None, metadata={"help": "Path to the data preprocessing config."}
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1, metadata={"help": "Number of processes to use for data preprocessing."}
    )
    skip_audio_processing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to skip the audio pre-preocessing stage when preparing the dataset."}
    )
    writer_batch_size: Optional[int] = field(default=500, metadata={"help": "Batch size to use for writing to disk."})
    text_transformations: Optional[List[str]] = field(
        default=None, metadata={"help": "List of transformations to apply to the text. "}
    )

    """Arguments defining structure of the dataset"""
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    context_text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data for datasets utilizing context. Defaults to 'text'. Currently only works when using a single dataset."},
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "Training split to be used."})
    validation_split: Optional[str] = field(default="validation", metadata={"help": "Validation split to be used."})
    test_splits: Optional[List[str]] = field(default=None, metadata={"help": "Splits to use for evaluation."})
    validation_slice: Optional[str] = field(default=None, metadata={"help": "Part of the validation split to be used."})
    sampling_rate: Optional[int] = field(default=16_000, metadata={"help": "Sampling rate for the model."})
    split_long_segments_to_chunks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to split long segments to chunks."}
    )
    cut_validation_from_train: Optional[bool] = field(
        default=False, metadata={"help": "Whether to cut validation split from train split."}
    )
    validation_slice_seed: Optional[int] = field(
        default=None, metadata={"help": "Seed to use for splitting validation slice."}
    )
    reshuffle_at_start: Optional[bool] = field(
        default=False, metadata={"help": "Whether to reshuffle the dataset at the start of preprocessing."}
    )
    pad_to_multiples_of: Optional[int] = field(
        default=None, metadata={"help": "Used in collator to pad to the multiples of x."}
    )
    dump_prepared_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path where to dump prepared datasets so it may be read preprocessed from single location."},
    )
    dataset_shard_size: Optional[str] = field(
        default=None, metadata={"help": "Size of the dataset shard to dump to disk."}
    )
    load_pure_dataset_only: Optional[bool] = field(
        default=False, metadata={"help": "Whether to load only the pure dataset without any preprocessing."}
    )
    how2_low_resource_split_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a file specifying utterance ids to keep in the how2 dataset."}
    )
    fisher_context_prefix: Optional[str] = field(
        default="Conversation context: ", metadata={"help": "Text prefix to the fisher conversation context."}
    )
    fisher_context_mode: Optional[str] = field(
        default="default", metadata={"help": "Fisher conversation data collator operation mode."}
    )
    fisher_max_context: Optional[int] = field(
        default=10, metadata={"help": "Fisher conversation context max length."}
    )
    fisher_context_trunc_to_shortest: Optional[bool] = field(
        default=False, metadata={"help": "Whether to truncate the fisher context to the shortest context length in the batch."}
    )
    slurp_use_slots: Optional[bool] = field(
        default=False, metadata={"help": "Whether to train for SLURP slot filling."}
    )
    slurp_dump_pred: Optional[bool] = field(
        default=False, metadata={"help": "Whether to dump all predictions into the wandb file.."}
    )
    do_not_remove_columns: Optional[List[str]] = field(
        default=None, metadata={"help": "List of column names to keep in the merged dataset."}
    )
    woz_use_agent_history: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use agent history in dialogue datasets."}
    )



@dataclass
class ModelArgumentsContext(ModelArguments):
    _argument_group_name = "Model-context specific arguments for context models."
    enc_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in encoder"}
    )
    dec_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in decoder"}
    )
    enc_memory_dim: int = field(default=None, metadata={"help": "Size of memory on encoder size"})

    dec_memory_dim: int = field(default=None, metadata={"help": "Size of memory on decoder size"})


@dataclass
class GeneralTrainingArgumentsContext(GeneralTrainingArguments):
    _argument_group_name = "Training-context specific arguments for context models."
    freeze_others: bool = field(default=False, metadata={"help": "Whether to freeze rest of the model"})
    conv_ids_column_name: str = field(default=None, metadata={"help": "Conv ids column."})
    turn_index_column_name: str = field(default=None, metadata={"help": "Turn index column."})


@dataclass
class TokenizerTrainingArguments:
    _argument_group_name = "Tokenizer related arguments"
    tokenizer_type: Optional[str] = field(
        default="unigram", metadata={"help": "Type of tokenizer to create if does not exists."}
    )
    additional_raw_data: Optional[str] = field(
        default=None,
        metadata={"help": "The input additional raw data file (a text file)."},
    )
    skip_if_exists: Optional[str] = field(default=None, metadata={"help": "Whether to check if tokenizer exists."})
    vocab_size: Optional[int] = field(default=5_000, metadata={"help": "Vocab size."})
    apply_regularization: Optional[bool] = field(default=False, metadata={"help": "Whether to apply regularization."})
    pad_token: Optional[str] = field(default="<pad>", metadata={"help": "PAD token"})
    mask_token: Optional[str] = field(default="<mask>", metadata={"help": "MASK token"})
    bos_token: Optional[str] = field(default="<s>", metadata={"help": "BOS token"})
    eos_token: Optional[str] = field(default="</s>", metadata={"help": "EOS token"})
    unk_token: Optional[str] = field(default="<unk>", metadata={"help": "UNK token"})

@dataclass
class ConnectorArguments:
    _argument_group_name = "QFormer alignment model arguments"
    qformer_config: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained qformer model/config or model/config identifier from huggingface.co/models"}
    )
    prompt_tuning_prefix_len: Optional[int] = field(default=0, metadata={"help": "Number of prefix embeddings for prompt tuning."})
    prompt_tuning_suffix_len: Optional[int] = field(default=0, metadata={"help": "Number of suffix embeddings for prompt tuning."})
    init_prompt_from_embeds: Optional[bool] = field(default=False, metadata={"help": "Whether to use the mean LLM embedding as the initalization point for the soft prompts (slow convergence)."})
    prompt_tuning_prefix_init: Optional[str] = field(default=None, metadata={"help": "Base text prompt prefix to initialize the soft prompt tuning."})
    prompt_tuning_suffix_init: Optional[str] = field(default=None, metadata={"help": "Base text prompt suffix to initialize the soft prompt tuning."})
    connector_type: Optional[str] = field(default='qformer', metadata={"help": "Architecture of the bridge network (qformer, conv)."})
    prompt_prefix: Optional[str] = field(default=None, metadata={"help": "Text prompt prefix to the connector output soft prompt."})
    prompt_suffix: Optional[str] = field(default=None, metadata={"help": "Text prompt suffix to the connector output soft prompt."})
    decoder_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to use LoRA for the decoder LM."})
    decoder_lora_rank: Optional[int] = field(default=8, metadata={"help": "Rank of the LoRA for the decoder LM."})
    decoder_lora_alpha: Optional[int] = field(default=8, metadata={"help": "Alpha of the LoRA for the decoder LM."})
    quantize_decoder: Optional[int] = field(default=None, metadata={"help": "Which BnB decoder quantization config to use (8bit, 4bit). FIXME: quant. order not working yet"})
    n_queries: Optional[int] = field(default=80, metadata={"help": "Number of qformer queries."})
    downsampling_factor: Optional[int] = field(default=4, metadata={"help": "When using the stacking downsampling method, concatenate 'N' consecutive embeddings."})
    conn_layers: Optional[int] = field(default=6, metadata={"help": "Number of qformer layers."})
    conn_hidden_size: Optional[int] = field(default=256, metadata={"help": "Qformer hidden dimension."})
    conn_attn_heads: Optional[int] = field(default=6, metadata={"help": "Number of qformer heads."})
    qf_intermediate_size: Optional[int] = field(default=2048, metadata={"help": "Qformer intermediate layer dimension."})
    qf_mm_pooling: Optional[str] = field(default=None, metadata={"help": "Modality-matching loss pooling method."})
    qf_mm_micro_loss: Optional[str] = field(default='dot', metadata={"help": "Modality-matching micro loss function."})
    qf_mm_loss_weight: Optional[float] = field(default=0.0, metadata={"help": "Modality-matching loss weight."})
    qf_config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default qformer config settings when a model is trained from scratch. Example: "
                "num_hidden_layers=4,hidden_size=256"
            )
        },
    )
