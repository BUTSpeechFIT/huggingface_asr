import glob
import os
import shutil
from typing import Dict

import torch
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    SequenceFeatureExtractor,
    Speech2TextFeatureExtractor,
    SpeechEncoderDecoderModel,
)
from transformers.utils import logging

from models.auto_wrappers import CustomAutoModelForCTC, CustomAutoModelForPretraining
from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
    JointCTCAttentionEncoderDecoderConfig,
)
from models.encoders.e_branchformer import (
    BestRQEBranchformerConfig,
    BestRQEBranchformerForPreTraining,
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
    Wav2Vec2EBranchformerForPreTraining,
)
from utilities.general_utils import average_dicts
from utilities.training_arguments import ModelArguments

logger = logging.get_logger("transformers")

AutoConfig.register("joint_aed_ctc_speech-encoder-decoder", JointCTCAttentionEncoderDecoderConfig)
AutoModelForSpeechSeq2Seq.register(JointCTCAttentionEncoderDecoderConfig, JointCTCAttentionEncoderDecoder)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
CustomAutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)
CustomAutoModelForPretraining.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForPreTraining)

AutoConfig.register("bestrq-ebranchformer", BestRQEBranchformerConfig)
CustomAutoModelForPretraining.register(BestRQEBranchformerConfig, BestRQEBranchformerForPreTraining)


def average_checkpoints(experiment_dir: str) -> str:
    checkpoints = glob.glob(f"{experiment_dir}/checkpoint*/pytorch_model.bin")
    state_dicts = [torch.load(checkpoint) for checkpoint in checkpoints]
    sum_state_dict, n_checkpoints = average_dicts(*state_dicts)
    del state_dicts
    average_dict = {key: value.div(n_checkpoints) for key, value in sum_state_dict.items()}
    dst_path = os.path.join(experiment_dir, "average_checkpoint")
    shutil.copytree(os.path.dirname(checkpoints[0]), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "tokenizer"), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "feature_extractor"), dst_path, dirs_exist_ok=True)
    torch.save(average_dict, os.path.join(dst_path, "pytorch_model.bin"))
    return dst_path


def fetch_config(
    enc_config_path: str, dec_config_path: str, base_config: Dict, config_overrides: str
) -> PretrainedConfig:
    enc_config = AutoConfig.from_pretrained(enc_config_path)
    dec_config = AutoConfig.from_pretrained(dec_config_path)
    config = JointCTCAttentionEncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    if config_overrides is not None:
        logger.info(f"Overriding config: {config_overrides}")
        parsed_dict = dict(x.split("=") for x in config_overrides.split(","))
        base_config.update(parsed_dict)
    kwargs_encoder = {
        argument[len("encoder_") :]: value for argument, value in base_config.items() if argument.startswith("encoder_")
    }
    kwargs_decoder = {
        argument[len("decoder_") :]: value
        for argument, value in base_config.items()
        if argument.startswith("decoder_") and argument != "decoder_start_token_id"
    }
    config.encoder.update(kwargs_encoder)
    config.decoder.update(kwargs_decoder)
    config.update(base_config)
    return config


def instantiate_ctc_model(
    model_args: ModelArguments, tokenizer: PreTrainedTokenizer, feature_extractor: SequenceFeatureExtractor
) -> PreTrainedModel:
    base_model_config = {
        "layerdrop": 0.0,
        "ctc_weight": model_args.ctc_weight,
        "ctc_loss_reduction": "mean",
        "vocab_size": len(tokenizer),
        "expect_2d_input": model_args.expect_2d_input,
    }
    if base_model_config["expect_2d_input"] and isinstance(feature_extractor, Speech2TextFeatureExtractor):
        base_model_config["second_dim_input_size"] = feature_extractor.num_mel_bins

    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = CustomAutoModelForCTC.from_pretrained(model_path, config=config)
    else:
        config = AutoConfig.from_pretrained(model_args.base_encoder_model)
        config.update(base_model_config)

        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            parsed_dict = dict(x.split("=") for x in model_args.config_overrides.split(","))
            config.update(parsed_dict)

        model = CustomAutoModelForCTC.from_config(config)

    return model


def instantiate_aed_model(
    model_args: ModelArguments, tokenizer: PreTrainedTokenizer, feature_extractor: SequenceFeatureExtractor
) -> SpeechEncoderDecoderModel:
    base_model_config = {
        "encoder_layerdrop": 0.0,
        "ctc_weight": model_args.ctc_weight,
        "encoder_ctc_loss_reduction": "mean",
        "pad_token_id": tokenizer.pad_token_id,
        "encoder_pad_token_id": tokenizer.pad_token_id,
        "encoder_vocab_size": len(tokenizer),
        "decoder_vocab_size": len(tokenizer),
        "lsm_factor": model_args.lsm_factor,
        "shared_lm_head": model_args.shared_lm_head,
        "encoder_expect_2d_input": model_args.expect_2d_input,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "decoder_pos_emb_fixed": model_args.decoder_pos_emb_fixed,
    }
    if base_model_config["encoder_expect_2d_input"] and isinstance(feature_extractor, Speech2TextFeatureExtractor):
        base_model_config["encoder_second_dim_input_size"] = feature_extractor.num_mel_bins

    # 4. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, config=config)
    elif model_args.from_encoder_decoder_config:
        config = fetch_config(
            model_args.base_encoder_model,
            model_args.base_decoder_model,
            base_model_config,
            model_args.config_overrides,
        )
        model = JointCTCAttentionEncoderDecoder(config=config)
    else:
        model = JointCTCAttentionEncoderDecoder.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=model_args.base_encoder_model,
            decoder_pretrained_model_name_or_path=model_args.base_decoder_model,
            **base_model_config,
        )
    return model


def instantiate_speech_encoder_model(
    model_args: ModelArguments, feature_extractor: SequenceFeatureExtractor
) -> PreTrainedModel:
    base_model_config = {
        "layerdrop": 0.0,
        "expect_2d_input": model_args.expect_2d_input,
    }
    if base_model_config["expect_2d_input"] and isinstance(feature_extractor, Speech2TextFeatureExtractor):
        base_model_config["second_dim_input_size"] = feature_extractor.num_mel_bins
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = CustomAutoModelForPretraining.from_pretrained(model_path, config=config)
    else:
        config = AutoConfig.from_pretrained(model_args.base_encoder_model)
        config.update(base_model_config)
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            parsed_dict = dict(x.split("=") for x in model_args.config_overrides.split(","))
            config.update(parsed_dict)
        model = CustomAutoModelForPretraining.from_config(config)
    return model
