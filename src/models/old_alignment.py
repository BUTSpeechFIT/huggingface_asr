"""Module containing all the preliminary alignment architecture definitions.

For up-to-date models, refer to the aligned.py module in this directiory.

Author: Simon Sedlacek
"""

from transformers import (

    BertModel,
    PreTrainedModel,
    PretrainedConfig,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Speech2TextModel,
    MarianMTModel,
    MarianForCausalLM,
    MarianConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, ModelOutput
from transformers.models.marian.modeling_marian import MarianEncoder
from tslearn.metrics import SoftDTWLossPyTorch

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from transformers.models.speech_to_text.modeling_speech_to_text import Conv1dSubsampler
from models.utils import SpeechEncoderOutputSubsampler, shift_tokens_right

@dataclass
class SpeechQFormerMarianOutput(ModelOutput):
    """
    TODO: rewrite
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    enc_loss: Optional[Tuple[torch.FloatTensor]] = None
    dec_loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    audio_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None


class  AlignmentConfig(PretrainedConfig):
    def __init__(
            self,
            encoder_config=None,
            qformer_config=None,
            connector_type='encoder_stacked',
            lm_config=None,
            lora_config=None,
            prompt=None,
            num_query_tokens=80,
            modality_matching=False,
            mm_pooling='avg',
            mm_micro_loss='dot',
            mm_loss_weight=1.0,
            ce_loss_weight=1.0,
            num_pretrain_epochs=0,
            downsampling_factor=6,
            prompt_tuning_prefix_len=0,
            prompt_tuning_suffix_len=0,
            init_prompt_from_embeds=False,
            prompt_tuning_prefix_init=None,
            prompt_tuning_suffix_init=None,
            freeze_encoder=True,
            **kwargs
        ):

        self.encoder_config = encoder_config
        if qformer_config is None:
            self.qformer_config = Blip2QFormerConfig()
        else:
            self.qformer_config = qformer_config

        self.lm_config = lm_config
        self.lora_config = lora_config
        # used to store the instruction prompt for the model (empty for most models, but used by some)
        self.prompt=None

        self.modality_matching = modality_matching
        self.mm_pooling=mm_pooling
        self.num_query_tokens = num_query_tokens
        self.mm_loss_weight = mm_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.num_pretrain_epochs = num_pretrain_epochs
        self.mm_micro_loss = mm_micro_loss
        self.connector_type = connector_type
        self.downsampling_factor = downsampling_factor
        self.prompt_tuning_prefix_len = prompt_tuning_prefix_len
        self.prompt_tuning_suffix_len = prompt_tuning_suffix_len
        self.init_prompt_from_embeds = init_prompt_from_embeds
        self.prompt_tuning_prefix_init = prompt_tuning_prefix_init
        self.prompt_tuning_suffix_init = prompt_tuning_suffix_init
        self.freeze_encoder = freeze_encoder

        super().__init__(**kwargs)

class ApmoModel(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    def __init__(
            self,
            pretrained_path: Optional[str] = None,
            config: Optional[AlignmentConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
        ):
        super().__init__(config)

        if encoder:
            #self.encoder = Speech2TextModel.from_pretrained(encoder).get_encoder()
            try:
                self.encoder = encoder.get_encoder()
            except:
                self.encoder = encoder.encoder
        else:
            raise ValueError("Encoder model needs to be supplied")
            #self.encoder = Speech2TextModel(config=config.encoder_config)

        if decoder:

            self.decoder = decoder

            try:
                self.mt_encoder = decoder.model.encoder
            except:
                self.mt_encoder = decoder.encoder

        else:
            raise ValueError("Decoder model needs to be supplied")
            #self.decoder = MarianMTModel(config=config.lm_config)

        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)
        self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
            )

        self.qformer = Blip2QFormerModel(config.qformer_config)

        if isinstance(config.lm_config, dict):
            if config.lm_config['architectures'][0] == 'MarianMTModel':
                config.lm_config = MarianConfig(**config.lm_config)
            else:
                raise ValueError("Unknown decoder class")

        if hasattr(config.lm_config, 'hidden_size'):
            lm_hidden_size = config.lm_config.hidden_size
        else:
            lm_hidden_size = config.lm_config.d_model

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, lm_hidden_size)

        # init the qformer self-attention with the mt_encoder self-attention
        if ('T5ForConditionalGeneration' not in self.config.lm_config.architectures and

            self.qformer.config.num_hidden_layers == self.mt_encoder.config.encoder_layers):
            for i in range(self.qformer.config.num_hidden_layers):
                self.qformer.encoder.layer[i].attention.attention.query.weight.data = self.mt_encoder.layers[i].self_attn.q_proj.weight.data
                self.qformer.encoder.layer[i].attention.attention.query.bias.data = self.mt_encoder.layers[i].self_attn.q_proj.bias.data
                self.qformer.encoder.layer[i].attention.attention.key.weight.data = self.mt_encoder.layers[i].self_attn.k_proj.weight.data
                self.qformer.encoder.layer[i].attention.attention.key.bias.data = self.mt_encoder.layers[i].self_attn.k_proj.bias.data
                self.qformer.encoder.layer[i].attention.attention.value.weight.data = self.mt_encoder.layers[i].self_attn.v_proj.weight.data
                self.qformer.encoder.layer[i].attention.attention.value.bias.data = self.mt_encoder.layers[i].self_attn.v_proj.bias.data

                self.qformer.encoder.layer[i].attention.output.dense.weight.data = self.mt_encoder.layers[i].self_attn.out_proj.weight.data
                self.qformer.encoder.layer[i].attention.output.dense.bias.data = self.mt_encoder.layers[i].self_attn.out_proj.bias.data

        # freeze encoder and decoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.mt_encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.decoder.lm_head.requires_grad = False
        if hasattr(self.decoder, 'final_logits_bias'):
            self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

        # fine-tuning-related flags
        self.encoder_ft = False
        self.decoder_ft = False
        self.pretraining = True if self.config.num_pretrain_epochs > 0 else False

        self.DIAG = 3
        self.diagonal_batched = torch.vmap(torch.diagonal)
        self.diag_embed_batched = torch.vmap(torch.diag_embed)
        #self.fs = []
        #for i in range(self.DIAG):
        #    self.fs.append(torch.vmap(lambda x: torch.diag_embed(self.diagonal_batched(x, offset=deepcopy(i)).view(x.shape[0], -1),offset=deepcopy(i))))

    def encoder_decoder_eval(self):
        # TODO: this will have to be modified when fine-tuning
        if not self.encoder_ft:
            self.encoder.eval()

        if not self.decoder_ft:
            self.decoder.eval()

    def unfreeze_decoder(self):
        self.decoder_ft = True
        for _, param in self.decoder.decoder.named_parameters():
            param.requires_grad = True
        self.decoder.lm_head.requires_grad = True

    def unfreeze_encoder(self):
        self.encoder_ft = True
        for _, param in self.encoder.named_parameters():
            param.requires_grad = True

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        mm_input_ids: Optional[torch.LongTensor] = None,
        mm_attention_mask: Optional[torch.LongTensor] = None,
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
            return_dict=return_dict
        )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # 2. pass the encoder output to the qformer
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = BaseModelOutput(
            last_hidden_state=self.language_projection(query_outputs[0]),
        )

        # compute the modality matching loss
        mm_loss = None
        if mm_input_ids is not None and (self.config.mm_loss_weight > 0.0 or self.pretraining):
            mt_encoder_output = self.mt_encoder(
                input_ids=mm_input_ids,
                attention_mask=mm_attention_mask,
                return_dict=return_dict,
            )

            if self.pretraining:
                # first, compute the macro loss
                query_pooled = torch.mean(query_output.last_hidden_state, 1)
                denom = torch.sum(mm_attention_mask, dim=1, keepdim=True)
                enc_pooled = torch.sum(mt_encoder_output[0], 1) / denom
                macro_loss = F.mse_loss(query_pooled, enc_pooled)

                # now compute the micro loss
                # l2 normalize the queries and mt embeddings
                query_l2 = F.normalize(query_output.last_hidden_state, dim=-1)
                mt_out_l2 = F.normalize(mt_encoder_output[0], dim=-1) * mm_attention_mask.unsqueeze(-1)
                if self.config.mm_micro_loss == 'dot':
                    dots = torch.bmm(query_l2, mt_out_l2.transpose(1, 2))
                    micro_loss = (1 - torch.max(dots, dim=-1).values).sum(dim=-1).mean()

                elif self.config.mm_micro_loss == 'dot_raw':
                    dots = torch.bmm(query_output.last_hidden_state,
                                     (mt_encoder_output[0] * mm_attention_mask.unsqueeze(-1)).transpose(1, 2)
                                     )
                    micro_loss = -(torch.max(dots, dim=-1).values).sum(dim=-1).mean()

                elif self.config.mm_micro_loss == 'dtw':
                    # try soft dtw
                    dtw = SoftDTWLossPyTorch(gamma=0.3, dist_func=lambda x, y: 1.0 - torch.bmm(x, y.transpose(1, 2)))
                    micro_loss = dtw(query_l2.float(), mt_out_l2.float()).mean()

                elif self.config.mm_micro_loss == 'new':
                    dots = torch.bmm(query_l2, mt_out_l2.transpose(1, 2))
                    # trim if |Q| < |E|
                    if query_l2.shape[1] < mt_out_l2.shape[1]:
                        print("warning, shapes", query_l2.shape, mt_out_l2.shape)
                        diff = mt_out_l2.shape[1] - query_l2.shape[1]
                        dots = dots[:,:,:]
                    else:
                        diff = query_l2.shape[1] - mt_out_l2.shape[1]
                        dots = dots[:,:-diff,:]

                    dots = 1.0 - dots
                    mask = 0
                    ones = torch.ones_like(dots)
                    for i in range(self.DIAG):
                        mask += torch.diag_embed(self.diagonal_batched(ones, offset=i).view(query_l2.shape[0], -1), offset=i)

                    sum = dots * mask
                    micro_loss = 10 * torch.mean(sum.sum(dim=2)/self.DIAG)

                else:
                    micro_loss = torch.tensor(0)

                mm_loss = macro_loss + micro_loss

            else:
                if self.config.mm_pooling == 'avg':

                    # now compute the micro loss
                    # l2 normalize the queries and mt embeddings
                    #query_l2 = F.normalize(query_output.last_hidden_state, dim=-1)
                    #mt_out_l2 = F.normalize(mt_encoder_output[0], dim=-1) * mm_attention_mask.unsqueeze(-1)
                    #query_l2 = query_output.last_hidden_state
                    #mt_out_l2 = mt_encoder_output[0] * mm_attention_mask.unsqueeze(-1)
                    #dots = torch.bmm(query_l2, mt_out_l2.transpose(1, 2))



                    query_pooled = torch.mean(query_output.last_hidden_state, 1)
                    denom = torch.sum(mm_attention_mask, dim=1, keepdim=True)
                    enc_pooled = torch.sum(mt_encoder_output[0], 1) / denom
                    mm_loss = F.mse_loss(query_pooled, enc_pooled)

                elif self.config.mm_pooling == 'max':
                    query_pooled = torch.max(query_output.last_hidden_state, 1)[0]
                    enc_pooled = torch.max(mt_encoder_output[0], 1)[0]
                    mm_loss = F.mse_loss(query_pooled, enc_pooled)

                elif self.config.mm_pooling == 'dot':
                    mat = torch.bmm(query_output.last_hidden_state, mt_encoder_output[0].transpose(1, 2))
                    mm_loss = -torch.max(mat)

        # return just the modality matching loss when pretraining the qformer
        if self.pretraining and self.training:
            return SpeechQFormerMarianOutput(
                loss=mm_loss,
                enc_loss=macro_loss.detach(),
                dec_loss=micro_loss.detach(),
                logits=None,
                audio_outputs=audio_embeds,
                qformer_outputs=None, #FIXME
                language_model_outputs=None, #FIXME
            )

        decoder_output = self.decoder(
            input_ids=None, # input_ids
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=query_output,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # combine the losses
        lm_loss = decoder_output.loss
        loss = None
        if mm_loss is None: mm_loss = 0

        loss = self.config.ce_loss_weight * lm_loss + self.config.mm_loss_weight * mm_loss

        return SpeechQFormerMarianOutput(
                loss=loss,
                enc_loss=mm_loss,
                dec_loss=decoder_output.loss,
                logits=decoder_output.logits,
                audio_outputs=audio_embeds,
                qformer_outputs=None, #FIXME
                language_model_outputs=None, #FIXME
            )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        mm_input_ids: Optional[torch.LongTensor] = None,
        mm_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        batch_size = input_features.shape[0]

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # 2. pass the encoder output to the qformer
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_attention_mask,
            return_dict=True,
        )
        query_output = BaseModelOutput(
            last_hidden_state=self.language_projection(query_outputs[0]),
        )

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(batch_size, 1).to(query_outputs[0].device)
        )

        attention_mask = torch.ones_like(decoder_input)
        decoder_output = self.decoder.generate(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=attention_mask,
            encoder_outputs=query_output,
            generation_config=self.generation_config,
            **generate_kwargs,
        )
        return decoder_output

class S2TEncoderMarianDecoder(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    def __init__(
            self,
            config: Optional[AlignmentConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
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

        ## freeze decoder encoder
        for _, param in self.decoder.model.encoder.named_parameters():
            param.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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

        decoder_output = self.decoder(
            input_ids=None,
            attention_mask=audio_attention_mask,
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        batch_size = input_features.shape[0]

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

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(batch_size, 1).to(audio_embeds.device)
        )

        attention_mask = torch.ones_like(decoder_input)
        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            attention_mask=audio_attention_mask,
            decoder_input_ids=decoder_input, # input_ids
            decoder_attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **generate_kwargs,
        )
        return decoder_output

class SpeechEncoderMarianEncoderDecoder(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[AlignmentConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
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

        conv_config = SpeechEncoderOutputSubsampler.default_config
        conv_config.update({'input_feat_per_channel': self.config.qformer_config.encoder_hidden_size})
        self.conv = SpeechEncoderOutputSubsampler(conv_config)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze decoder
        for _, param in self.decoder.model.decoder.named_parameters():
            param.requires_grad = False
        self.decoder.lm_head.requires_grad = False
        self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def encoder_decoder_eval(self):
        # TODO: this will have to be modified when fine-tuning
        self.encoder.eval()
        self.decoder.model.decoder.eval()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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

        # Downsample the speech encoder outputs
        encoder_outputs.last_hidden_state = self.conv(encoder_outputs.last_hidden_state)

        # downsample encoder attention mask again..
        if audio_attention_mask is not None:
            audio_attention_mask = self.conv._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], audio_attention_mask
            )
        else:
            audio_attention_mask = None

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=encoder_outputs.last_hidden_state,
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
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        batch_size = input_features.shape[0]

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

        # Downsample the speech encoder outputs
        audio_embeds = self.conv(audio_embeds)

        # downsample encoder attention mask again..
        if audio_attention_mask is not None:
            audio_attention_mask = self.conv._get_feature_vector_attention_mask(
                audio_embeds.shape[1], audio_attention_mask
            )
        else:
            audio_attention_mask = None

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            inputs_embeds=audio_embeds,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )
        return decoder_output


class SpeechEncoderConvMarianEncoderDecoder(PreTrainedModel):
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

        conv_config = SpeechEncoderOutputSubsampler.default_config
        conv_config.update({'input_feat_per_channel': self.config.qformer_config.encoder_hidden_size})
        self.conv = SpeechEncoderOutputSubsampler(conv_config)

        q_conf = self.config.qformer_config
        marian_config = deepcopy(self.decoder.config)
        marian_config.update({
            "d_model": q_conf.hidden_size,
            "encoder_layers": q_conf.num_hidden_layers,
            "encoder_attention_heads": q_conf.num_attention_heads,
            "encoder_ffn_dim": q_conf.intermediate_size,
        })

        self.connector = MarianEncoder(marian_config)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze whole decoder
        for _, param in self.decoder.model.encoder.named_parameters():
            param.requires_grad = False

        self.freeze_decoder = freeze_decoder
        if freeze_decoder:
            for _, param in self.decoder.model.decoder.named_parameters():
                param.requires_grad = False
            self.decoder.lm_head.requires_grad = False
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

        # Downsample the speech encoder outputs
        encoder_outputs.last_hidden_state = self.conv(encoder_outputs.last_hidden_state)

        # downsample encoder attention mask again..
        if audio_attention_mask is not None:
            audio_attention_mask = self.conv._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], audio_attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs = self.connector(
                attention_mask=audio_attention_mask,
                inputs_embeds=encoder_outputs.last_hidden_state,
                return_dict=True,
        )

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
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        batch_size = input_features.shape[0]

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

        # Downsample the speech encoder outputs
        audio_embeds = self.conv(audio_embeds)

        # downsample encoder attention mask again..
        if audio_attention_mask is not None:
            audio_attention_mask = self.conv._get_feature_vector_attention_mask(
                audio_embeds.shape[1], audio_attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs = self.connector(
                attention_mask=audio_attention_mask,
                inputs_embeds=audio_embeds,
                return_dict=True,
        )

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            inputs_embeds=connector_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )
        return decoder_output
