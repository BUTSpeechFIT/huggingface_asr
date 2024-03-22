import glob
import json
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
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.utils import logging

from decoding.config import GenerationConfigCustom
from models.auto_wrappers import (
    CustomAutoModelForCTC,
    CustomAutoModelForPretraining,
    CustomModelForCausalLM,
)
from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
    JointCTCAttentionEncoderDecoderConfig,
)
from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from models.decoders.multi_head_gpt2_mixing import (
    GPT2LMMultiHeadModelMixing,
    GPT2MultiHeadMixingConfig,
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


def fetch_config(config: PretrainedConfig, base_config: Dict, config_overrides: str) -> PretrainedConfig:
    if config_overrides is not None:
        logger.info(f"Overriding config: {config_overrides}")
        parsed_dict = base_config | dict(x.split("=") for x in config_overrides.split(";"))
    else:
        parsed_dict = base_config
    for k_orig, v in parsed_dict.items():
        if k_orig.startswith("encoder_"):
            config_local = config.encoder
            k = k_orig[len("encoder_") :]
        elif k_orig.startswith("decoder_") and k_orig != "decoder_start_token_id":
            config_local = config.decoder
            k = k_orig[len("decoder_") :]
        else:
            config_local = config
            k = k_orig
        if not hasattr(config_local, k):
            if k_orig in base_config:
                setattr(config_local, k, type(base_config[k_orig])(v))
            else:
                raise ValueError(f"key {k} isn't in the original config dict")

        old_v = getattr(config_local, k)
        if isinstance(old_v, bool):
            if isinstance(v, bool):
                setattr(config_local, k, v)
                continue
            if v.lower() in ["true", "1", "y", "yes"]:
                v = True
            elif v.lower() in ["false", "0", "n", "no"]:
                v = False
            else:
                raise ValueError(f"can't derive true or false from {v} (key {k})")
        elif isinstance(old_v, int):
            v = int(v)
        elif isinstance(old_v, float):
            v = float(v)
        elif isinstance(old_v, list):
            v = json.loads(v)
        elif old_v is None:
            pass
        elif not isinstance(old_v, str):
            raise ValueError(
                f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
            )
        setattr(config_local, k, v)
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
        config = fetch_config(config, base_model_config, model_args.config_overrides)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, config=config)
    elif model_args.from_encoder_decoder_config:
        enc_config = AutoConfig.from_pretrained(model_args.base_encoder_model)
        dec_config = AutoConfig.from_pretrained(model_args.base_decoder_model)
        config = JointCTCAttentionEncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
        config = fetch_config(
            config,
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
    if model_args.finetune_mixing_mechanism:
        if not isinstance(model.decoder, GPT2LMMultiHeadModel):
            raise ValueError("The model decoder must be an instance of GPT2LMMultiHeadModel")
        old_config = model.decoder.config
        new_config = GPT2MultiHeadMixingConfig(**old_config.to_dict(), mixing_mode=model_args.finetune_mixing_mechanism)
        new_decoder = CustomModelForCausalLM.from_config(new_config)
        new_decoder.load_state_dict(model.decoder.state_dict(), strict=False)

        model.decoder = new_decoder
        # Freeze the weights of the model decoder, except for the mixing mechanism
        for name, param in model.named_parameters():
            if "lm_mixing" not in name:
                param.requires_grad = False
        logger.info("Reinstantiating the model decoder with the model with the model supporting mixing mechanism.")
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
            config.update_from_string(model_args.config_overrides)
        model = CustomAutoModelForPretraining.from_config(config)
    return model


def handle_whisper_generation_config(
    model_args: ModelArguments,
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperTokenizer,
    gen_config: GenerationConfigCustom,
):
    if model_args.whisper_task and model_args.whisper_language:
        gen_config.suppress_tokens = []
        gen_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
            language=model_args.whisper_language, task=model_args.whisper_task
        )
    gen_config.max_length = model.generation_config.max_length
    gen_config.decoder_start_token_id = model.generation_config.decoder_start_token_id
