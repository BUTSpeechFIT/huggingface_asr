"""Module containing all the up-to-date alignment architecture definitions.

Both the ECD and ECED architectures are confirmed to be compatible with
the E-Branchformer/Whisper encoders and T5/MarianMT decoders.

Author: Simon Sedlacek
"""

from transformers import (
    PreTrainedModel,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    MarianConfig,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.marian.modeling_marian import MarianEncoder

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from models.utils import SpeechEncoderOutputSubsampler, shift_tokens_right

from models.old_alignment import SpeechQFormerMarianOutput, AlignmentConfig

# define the forward method wrapper
def soft_prompt_wrapper(func):
    def inner_wrapper(self, encoder_outputs, attention_mask):
        encoder_outputs, attention_mask = func(self, encoder_outputs, attention_mask)
        batch_size = encoder_outputs.last_hidden_state.shape[0]

        if self.prompt_prefix is not None:
            prefix_expanded = self.prompt_prefix.expand(batch_size, -1, -1)

            # prepend the soft prefix to the embeddings
            encoder_outputs.last_hidden_state = torch.cat((prefix_expanded, encoder_outputs.last_hidden_state), dim=1)

            # extend the attention mask -- do not disrupt the original one due to padding
            if attention_mask is not None:
                attn_mask_prefix = torch.ones(
                    prefix_expanded.shape[:2],
                    device=attention_mask.device,
                    dtype=torch.long,
                )
                attention_mask = torch.cat((attn_mask_prefix, attention_mask), dim=1)

        if self.prompt_suffix is not None:
            suffix_expanded = self.prompt_suffix.expand(batch_size, -1, -1)

            # append the soft suffix to the embeddings
            encoder_outputs.last_hidden_state = torch.cat((encoder_outputs.last_hidden_state, suffix_expanded), dim=1)

            # extend the attention mask -- do not disrupt the original one due to padding
            if attention_mask is not None:
                attn_mask_suffix = torch.ones(
                    suffix_expanded.shape[:2],
                    device=attention_mask.device,
                    dtype=torch.long,
                )
                attention_mask = torch.cat((attention_mask, attn_mask_suffix), dim=1)

        return encoder_outputs, attention_mask
    return inner_wrapper

class AlignmentNetwork(PreTrainedModel):
    config_class = AlignmentConfig

    def __init__(
            self,
            config: AlignmentConfig,
            model_type: Optional[str] = 'qformer',
            soft_prompt_init: Optional[torch.Tensor] = None,
            init_prefix_from: Optional[torch.Tensor] = None,
            init_suffix_from: Optional[torch.Tensor] = None, # TODO: embeddings or token ids? (for ids we need emb.layer)
            tokenizer: Optional[PreTrainedTokenizer] = None,
        ):
        super().__init__(config)
        self.model_type = model_type

        # fix the configs if needed..
        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)

        if hasattr(config.lm_config, 'hidden_size'):
            lm_hidden_size = config.lm_config.hidden_size
        else:
            lm_hidden_size = config.lm_config.d_model

        # first define the language projection as it's used by both alignment configurations
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, lm_hidden_size)

        # set up soft prompts
        if config.prompt_tuning_prefix_len > 0 or init_prefix_from is not None:

            if soft_prompt_init is not None:
                self.prompt_prefix = nn.Parameter(
                    torch.ones(1, config.prompt_tuning_prefix_len, lm_hidden_size) * torch.squeeze(soft_prompt_init)
                )

            elif init_prefix_from is not None:
                if tokenizer is not None:
                    raise NotImplementedError

                else: # assume prompt embeddings were passed
                    assert len(init_prefix_from.shape) == 3, "We want shape of [1, N, D]"
                    assert init_prefix_from.shape[-1] == lm_hidden_size, "Incorrect prompt embedding dimension"
                    self.prompt_prefix = nn.Parameter(init_prefix_from)

            else: # init to zeros..
                self.prompt_prefix = nn.Parameter(
                    torch.zeros(1, config.prompt_tuning_prefix_len, lm_hidden_size)
                )
        else: self.prompt_prefix = None

        if config.prompt_tuning_suffix_len > 0 or init_suffix_from is not None:

            if soft_prompt_init is not None:
                self.prompt_suffix = nn.Parameter(
                    torch.ones(1, config.prompt_tuning_suffix_len, lm_hidden_size) * torch.squeeze(soft_prompt_init)
                )

            elif init_suffix_from is not None:
                if tokenizer is not None:
                    raise NotImplementedError

                else: # assume prompt embeddings were passed
                    assert len(init_suffix_from.shape) == 3, "We want shape of [1, N, D]"
                    assert init_suffix_from.shape[-1] == lm_hidden_size, "Incorrect prompt embedding dimension"
                    self.prompt_suffix = nn.Parameter(init_suffix_from)

            else: # init to zeros..
                self.prompt_suffix = nn.Parameter(
                    torch.zeros(1, config.prompt_tuning_suffix_len, lm_hidden_size)
                )

        else: self.prompt_suffix = None

        # setup the connector type...
        if self.model_type == 'qformer':

            self.query_tokens = nn.Parameter(
                    torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
                )

            self.qformer = Blip2QFormerModel(config.qformer_config)
            self.forward = self._forward_qformer

        elif self.model_type == 'linear':

            conv_config = SpeechEncoderOutputSubsampler.default_config
            conv_config.update({
                'input_feat_per_channel': config.qformer_config.encoder_hidden_size,
                'd_model': config.qformer_config.hidden_size,
                })
            self.conv = SpeechEncoderOutputSubsampler(conv_config)

            self.fc = nn.Linear(config.qformer_config.hidden_size, config.qformer_config.hidden_size)
            self.forward = self._forward_linear

        elif self.model_type == 'linear_stacked':

            downsampling_factor = config.downsampling_factor
            self.fc = nn.Linear(config.qformer_config.encoder_hidden_size * downsampling_factor, config.qformer_config.hidden_size)
            self.attention_pooling = nn.MaxPool1d(downsampling_factor, stride=downsampling_factor)
            self.forward = self._forward_linear_stacked

        elif self.model_type == 'ste':
            conv_config = SpeechEncoderOutputSubsampler.default_config
            conv_config.update({
                'input_feat_per_channel': config.qformer_config.encoder_hidden_size,
                'd_model': config.qformer_config.hidden_size,
                })
            self.conv = SpeechEncoderOutputSubsampler(conv_config)

            q_conf = config.qformer_config
            marian_config = MarianConfig(
                    d_model=q_conf.hidden_size,
                    encoder_layers=q_conf.num_hidden_layers,
                    decoder_layers=q_conf.num_hidden_layers,
                    encoder_attention_heads=q_conf.num_attention_heads,
                    decoder_attention_heads=q_conf.num_attention_heads,
                    encoder_ffn_dim=q_conf.intermediate_size,
                    decoder_ffn_dim=q_conf.intermediate_size,
                    activation_function='gelu_new',
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    scale_embedding=True,
                    )

            self.connector = MarianEncoder(marian_config)
            self.connector.embed_tokens.weight.requires_grad = False
            self.forward = self._forward_ste

        elif self.model_type == 'encoder_stacked':
            downsampling_factor = config.downsampling_factor
            self.fc = nn.Linear(config.qformer_config.encoder_hidden_size * downsampling_factor, config.qformer_config.hidden_size)
            self.attention_pooling = nn.MaxPool1d(downsampling_factor, stride=downsampling_factor)

            q_conf = config.qformer_config
            marian_config = MarianConfig(
                    d_model=q_conf.hidden_size,
                    encoder_layers=q_conf.num_hidden_layers,
                    decoder_layers=q_conf.num_hidden_layers,
                    encoder_attention_heads=q_conf.num_attention_heads,
                    decoder_attention_heads=q_conf.num_attention_heads,
                    encoder_ffn_dim=q_conf.intermediate_size,
                    decoder_ffn_dim=q_conf.intermediate_size,
                    activation_function='gelu_new',
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    scale_embedding=True,
                    )

            self.connector = MarianEncoder(marian_config)
            self.connector.embed_tokens.weight.requires_grad = False
            self.forward = self._forward_encoder_stacked


    @soft_prompt_wrapper
    def _forward_qformer(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs,
                )

        audio_embeds = encoder_outputs.last_hidden_state

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_output = BaseModelOutput(
            last_hidden_state=self.language_projection(query_outputs[0]),
        )

        return query_output, None

    @soft_prompt_wrapper
    def _forward_linear(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs,
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # Downsample the speech encoder outputs
        audio_embeds = self.conv(audio_embeds)

        # downsample encoder attention mask again..
        if attention_mask is not None:
            attention_mask = self.conv._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            attention_mask = None

        # pass the encoder outputs through the bridge network
        fc_out = F.gelu(self.fc(audio_embeds))

        connector_outputs = BaseModelOutput(
                last_hidden_state = self.language_projection(fc_out)
            )

        return connector_outputs, attention_mask

    @soft_prompt_wrapper
    def _forward_linear_stacked(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        audio_embeds = encoder_outputs.last_hidden_state

        # Downsample the speech encoder outputs
        downsampling_factor = self.config.downsampling_factor
        mod = audio_embeds.shape[-2] % downsampling_factor
        if mod != 0:
            # append zeros to both the embeddings and the mask if the sequences are not divisible
            # by downsampling_factor
            appendix = torch.zeros((audio_embeds.shape[0], mod, audio_embeds.shape[-1]))
            audio_embeds = torch.hstack((audio_embeds, appendix))

            if attention_mask is not None:
                mask_appendix = attention_mask[...,-1].unsqueeze(1).repeat(1, mod)
                attention_mask = torch.cat((attention_mask, mask_appendix), dim=1)

        # perform the stacking downsampling
        audio_embeds = audio_embeds.contiguous().view(
            audio_embeds.shape[0],
            audio_embeds.shape[1] // downsampling_factor,
            audio_embeds.shape[2] * downsampling_factor
        )

        #embs = []
        #for i in range(downsampling_factor):
        #    embs.append(audio_embeds[...,i::downsampling_factor,:])

        #audio_embeds = torch.cat(embs, dim=-1)

        # downsample the attention mask too
        if attention_mask is not None:
            attention_mask = self.attention_pooling(attention_mask.float()).long()
        else:
            attention_mask = None

        # project the downsampled embeddings through the linear layers..
        fc_out = F.gelu(self.fc(audio_embeds))

        connector_outputs = BaseModelOutput(
                last_hidden_state = self.language_projection(fc_out)
            )

        return connector_outputs, attention_mask

    @soft_prompt_wrapper
    def _forward_ste(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs,
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # Downsample the speech encoder outputs
        audio_embeds = self.conv(audio_embeds)

        # downsample encoder attention mask again..
        if attention_mask is not None:
            attention_mask = self.conv._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs = self.connector(
                attention_mask=attention_mask,
                inputs_embeds=audio_embeds,
                return_dict=True,
        )

        connector_outputs.last_hidden_state = self.language_projection(connector_outputs.last_hidden_state)

        return connector_outputs, attention_mask

    @soft_prompt_wrapper
    def _forward_encoder_stacked(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        audio_embeds = encoder_outputs.last_hidden_state

        # Downsample the speech encoder outputs
        downsampling_factor = self.config.downsampling_factor
        mod = audio_embeds.shape[-2] % downsampling_factor
        if mod != 0:
            # append zeros to both the embeddings and the mask if the sequences are not divisible
            # by downsampling_factor
            appendix = torch.zeros((audio_embeds.shape[0], downsampling_factor - mod,
                                    audio_embeds.shape[-1]), device=audio_embeds.device)
            audio_embeds = torch.hstack((audio_embeds, appendix))

            if attention_mask is not None:
                mask_appendix = attention_mask[...,-1].unsqueeze(1).repeat(1, downsampling_factor - mod)
                attention_mask = torch.cat((attention_mask, mask_appendix), dim=1)

        # perform the stacking downsampling
        audio_embeds = audio_embeds.contiguous().view(
            audio_embeds.shape[0],
            audio_embeds.shape[1] // downsampling_factor,
            audio_embeds.shape[2] * downsampling_factor
        )

        #embs = []
        #for i in range(downsampling_factor):
        #    embs.append(audio_embeds[...,i::downsampling_factor,:])

        #audio_embeds = torch.cat(embs, dim=-1)

        # downsample the attention mask too
        if attention_mask is not None:
            attention_mask = self.attention_pooling(attention_mask.float()).long()
        else:
            attention_mask = None

        # project the downsampled embeddings through the linear layers..
        # NOTE: I removed the gelu activation as it didn't seem right to place
        # a nonlinearity before the transformer encoder
        audio_embeds = self.fc(audio_embeds)

        # pass the encoder outputs through the bridge network
        connector_outputs = self.connector(
                attention_mask=attention_mask,
                inputs_embeds=audio_embeds,
                return_dict=True,
        )

        connector_outputs.last_hidden_state = self.language_projection(connector_outputs.last_hidden_state)

        return connector_outputs, attention_mask



class SpeechEncoderBridgeTextDecoder(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[AlignmentConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            freeze_decoder: Optional[bool] = True,
        ):
        super().__init__(config)

        if encoder:
            try:
                self.encoder = encoder.get_encoder()
            except:
                self.encoder = encoder.encoder
        else:
            raise ValueError("Encoder model needs to be supplied")

        if decoder:
            self.decoder = decoder

        else:
            raise ValueError("Decoder model needs to be supplied")

        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)
        if isinstance(config.lm_config, dict):
            config.lm_config = PretrainedConfig(**config.lm_config)

        self.connector = AlignmentNetwork(config=self.config, model_type=self.config.connector_type)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze whole decoder
        if hasattr(self.decoder, 'model'):
            for _, param in self.decoder.model.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.decoder.model.decoder.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.decoder.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.decoder.decoder.named_parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.decoder.lm_head.requires_grad = False
        if hasattr(self.decoder, 'final_logits_bias'):
            self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def encoder_decoder_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SpeechQFormerMarianOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        """
        if prompt_prefix_ids is not None:

            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(prompt_prefix_ids)

            connector_outputs.last_hidden_state = torch.hstack((prefix_embeds, connector_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((prompt_prefix_mask, audio_attention_mask))
        """

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=audio_attention_mask,
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=connector_outputs,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return decoder_output

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        # maintain comaptability with marian
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
                encoder_outputs=audio_embeds,
                attention_mask=audio_attention_mask,
            )

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(audio_embeds.shape[0], 1).to(audio_embeds.device)
        )

        attention_mask = torch.ones_like(decoder_input)

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            attention_mask=audio_attention_mask,
            encoder_outputs=connector_outputs,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )
        return decoder_output


class SpeechEncoderBridgeMarianEncoderDecoder(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[AlignmentConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            freeze_decoder: Optional[bool] = True,
        ):
        super().__init__(config)

        if encoder:
            try:
                self.encoder = encoder.get_encoder()
            except:
                self.encoder = encoder.encoder
        else:
            raise ValueError("Encoder model needs to be supplied")

        if decoder:
            self.decoder = decoder

        else:
            raise ValueError("Decoder model needs to be supplied")

        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)
        if isinstance(config.lm_config, dict):
            config.lm_config = PretrainedConfig(**config.lm_config)

        self.connector = AlignmentNetwork(config=self.config, model_type=self.config.connector_type)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze whole decoder
        if hasattr(self.decoder, 'model'):
            for _, param in self.decoder.model.encoder.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.decoder.encoder.named_parameters():
                param.requires_grad = False

        self.freeze_decoder = freeze_decoder
        if freeze_decoder:
            if hasattr(self.decoder, 'model'):
                for _, param in self.decoder.model.decoder.named_parameters():
                    param.requires_grad = False
            else:
                for _, param in self.decoder.decoder.named_parameters():
                    param.requires_grad = False
                for param in self.decoder.parameters():
                    param.requires_grad = False

            if hasattr(self.decoder, 'lm_head'):
                self.decoder.lm_head.requires_grad = False
            if hasattr(self.decoder, 'final_logits_bias'):
                self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def encoder_decoder_eval(self):
        # TODO: this will have to be modified when fine-tuning
        self.encoder.eval()
        if self.freeze_decoder:
            self.decoder.eval()
        else:
            self.decoder.model.encoder.eval()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SpeechQFormerMarianOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        if prompt_prefix_ids is not None:


            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(prompt_prefix_ids)

            connector_outputs.last_hidden_state = torch.hstack((prefix_embeds, connector_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((prompt_prefix_mask, audio_attention_mask))

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=connector_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=None,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return decoder_output

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        # maintain comaptability with marian
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
                encoder_outputs=audio_embeds,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        if prompt_prefix_ids is not None:

            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(prompt_prefix_ids)

            connector_outputs.last_hidden_state = torch.hstack((prefix_embeds, connector_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((prompt_prefix_mask, audio_attention_mask))

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            inputs_embeds=connector_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )
        return decoder_output

