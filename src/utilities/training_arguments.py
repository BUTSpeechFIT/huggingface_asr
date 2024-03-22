from dataclasses import dataclass, field
from typing import List, Optional, Union

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
    from_encoder_decoder_config: Optional[bool] = field(
        default=False, metadata={"help": "Whether to create model from encoder and decoder configs."}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
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


@dataclass
class GeneralTrainingArguments(Seq2SeqTrainingArguments):
    _argument_group_name = "Training related arguments"
    """Arguments related to phases of the training."""
    preprocess_dataset_only: bool = field(default=False, metadata={"help": "Whether to preprocess dataset only"})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})
    do_evaluate: Optional[bool] = field(default=False, metadata={"help": "Whether to run evaluation."})
    do_generate: Optional[bool] = field(default=False, metadata={"help": "Whether to run generation."})
    restart_from: Optional[str] = field(
        default="", metadata={"help": "Path to checkpoint used to restart the training."}
    )

    """Preprocessing and postprocessing related arguments."""
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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
    mask_unks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to mask unknown tokens for cross entropy."}
    )


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
    post_process_predicitons: Optional[bool] = field(
        default=False, metadata={"help": "Whether to post process predictions."}
    )
    apply_eos_space_trick: Optional[bool] = field(default=False, metadata={"help": "Whether to apply eos space trick."})
    eos_space_trick_weight: Optional[float] = field(default=0.0, metadata={"help": "Weight of eos space trick."})
    space_token_id: Optional[int] = field(default=-1, metadata={"help": "Space token id."})


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
