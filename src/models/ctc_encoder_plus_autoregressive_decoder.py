from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    SpeechEncoderDecoderConfig,
    SpeechEncoderDecoderModel,
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutput, Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)
from transformers.utils import logging

from decoding.ctc_scorer import CTCRescorerLogitsProcessor, LogSoftmaxProcessor
from decoding.shallow_fussion import LMRescorerLogitsProcessor
from models.auto_wrappers import CustomAutoModelForCTC, CustomModelForCausalLM
from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from models.decoders.multi_head_gpt2_mixing import (
    GPT2LMMultiHeadModelMixing,
    GPT2MultiHeadMixingConfig,
)
from models.decoders.residual_clasiffier_gpt2 import (
    GPT2ResidualsLMHeadConfig,
    GPT2ResidualsLMHeadModel,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
)

logger = logging.get_logger("transformers")

AutoConfig.register("gpt2-multi-head", GPT2MultiHeadConfig)
CustomModelForCausalLM.register(GPT2MultiHeadConfig, GPT2LMMultiHeadModel)

AutoConfig.register("gpt2-multi-head-mixing", GPT2MultiHeadMixingConfig)
CustomModelForCausalLM.register(GPT2MultiHeadMixingConfig, GPT2LMMultiHeadModelMixing)

AutoConfig.register("gpt2-residuals-head", GPT2ResidualsLMHeadConfig)
CustomModelForCausalLM.register(GPT2ResidualsLMHeadConfig, GPT2ResidualsLMHeadModel)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
CustomAutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)


def wav2vec2_forward_hidden_return_hook(_: PreTrainedModel, __: Any, kwargs):
    kwargs["output_hidden_states"] = True


class JointCTCAttentionEncoderDecoderConfig(SpeechEncoderDecoderConfig):
    model_type = "joint_aed_ctc_speech-encoder-decoder"
    is_composition = True


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None
    encoder_logits: Optional[torch.FloatTensor] = None


def wav2vec2_for_ctc_forward_hook(model: CustomAutoModelForCTC, input: Any, output: CausalLMOutput):
    if "hidden_states" in output:
        output.last_hidden_state = output.hidden_states[-1]


class JointCTCAttentionEncoderDecoder(SpeechEncoderDecoderModel):
    """Custom model for CTC+Attention loss based on the ESPNet architecture"""

    config_class = JointCTCAttentionEncoderDecoderConfig
    base_model_prefix = "joint_aed_ctc_speech-encoder-decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

            # initialize with config
            # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super(SpeechEncoderDecoderModel, self).__init__(config)

        if encoder is None:
            encoder = CustomAutoModelForCTC.from_config(config.encoder)
            encoder.register_forward_hook(wav2vec2_for_ctc_forward_hook)
            encoder.register_forward_pre_hook(wav2vec2_forward_hidden_return_hook, with_kwargs=True)
        if decoder is None:
            decoder = CustomModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )
        self.enc_loss_weight = config.ctc_weight
        self.dec_loss_weight = 1 - config.ctc_weight
        self.lsm_factor = config.lsm_factor

        if config.shared_lm_head:
            self.encoder.lm_head.weight = self.decoder.lm_head.weight

        self.encoder_logits = None
        self.encoder_output_lens = None

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_") and argument != "decoder_start_token_id"
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = CustomAutoModelForCTC.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )
            encoder.register_forward_hook(wav2vec2_for_ctc_forward_hook)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = CustomModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = JointCTCAttentionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputLosses]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = CausalLMOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs.last_hidden_state

        # optionally project encoder_hidden_states
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if hasattr(self.decoder, "head_weights") and len(self.decoder.head_weights) > 1
            else output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = enc_loss = dec_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=self.lsm_factor)
            enc_loss = encoder_outputs.loss if return_dict else encoder_outputs[0]
            dec_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            dec_loss = loss_fct(dec_logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            loss = self.enc_loss_weight * enc_loss + self.dec_loss_weight * dec_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputLosses(
            loss=loss,
            enc_loss=enc_loss,
            dec_loss=dec_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_logits=encoder_outputs.logits,
        )

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # pylint: disable=no-member
        processors = super()._get_logits_processor(
            generation_config,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            model_kwargs,
            negative_prompt_ids,
            negative_prompt_attention_mask,
        )
        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            if generation_config.num_beams <= 1:
                processors.append(LogSoftmaxProcessor())
            self.ctc_rescorer = CTCRescorerLogitsProcessor(
                self.encoder_logits,
                self.encoder_output_lens,
                self.generation_config.pad_token_id,
                self.generation_config.eos_token_id,
                self.generation_config.ctc_margin,
                self.generation_config.ctc_weight,
                self.generation_config.num_beams,
                self.generation_config.space_token_id,
                self.generation_config.apply_eos_space_trick,
                self.generation_config.eos_space_trick_weight,
            )
            processors.append(self.ctc_rescorer)
        if hasattr(generation_config, "lm_weight") and generation_config.lm_weight > 0:
            if not hasattr(generation_config, "lm_model"):
                raise ValueError("If `lm_weight` is specified, make sure that `lm_model` is defined.")
            processors.append(
                LMRescorerLogitsProcessor(generation_config.lm_weight, generation_config.lm_model, device=self.device)
            )
        return processors

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        self.encoder_output_lens = self.encoder._get_feat_extract_output_lengths(
            model_kwargs["attention_mask"].sum(dim=1)
        )

        # pylint: disable=no-member
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )
        self.encoder_logits = model_kwargs["encoder_outputs"].logits
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor) and key != "loss":
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
            model_kwargs["encoder_outputs"].last_hidden_state = model_kwargs[
                "encoder_outputs"
            ].last_hidden_state.repeat_interleave(expand_size, dim=0)

        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if "encoder_outputs" in kwargs:
            self.encoder_logits = kwargs["encoder_outputs"].logits
            self.encoder_output_lens = self.encoder._get_feat_extract_output_lengths(
                kwargs["attention_mask"].sum(dim=1)
            )
        # pylint: disable=no-member
        output = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            **kwargs,
        )
        self.encoder_logits = None
        self.encoder_output_lens = None
        return output
