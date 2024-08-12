"""
This module implements a CTC-based model for speech recognition using the Whisper model.
"""
import copy
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.whisper.modeling_whisper import (
    ACT2FN,
    WHISPER_ATTENTION_CLASSES,
    WhisperConfig,
    WhisperEncoder,
    WhisperEncoderLayer,
    WhisperPreTrainedModel,
)


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class CustomWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.out_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WhisperForCTCConfig(WhisperConfig):
    """This is a modified version of the `WhisperEncoder` model from the `transformers` library.
    The model has been modified to support CTC loss computation in the forward pass."""

    def __init__(
        self,
        ctc_loss_reduction: str = "mean",
        final_dropout: float = 0.0,
        ctc_zero_infinity: bool = False,
        ctc_weight: float = 0.0,
        blank_token_id: Optional[int] = None,
        additional_layer: bool = False,
        additional_self_attention_layer: bool = False,
        sub_sample: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.final_dropout = final_dropout
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_weight = ctc_weight
        self.blank_token_id = blank_token_id
        self.additional_layer = additional_layer
        self.additional_self_attention_layer = additional_self_attention_layer
        self.sub_sample = sub_sample


_HIDDEN_STATES_START_POSITION = 2


class WhisperEncoderForCTC(WhisperPreTrainedModel):
    config_class = WhisperForCTCConfig

    def __init__(self, config: WhisperForCTCConfig, llm_dim):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)

        extended_config = copy.deepcopy(config)
        extended_config.d_model = llm_dim
        extended_config.encoder_ffn_dim = llm_dim * 4
        extended_config.encoder_attention_heads = 8
        self.dim_matching = nn.Linear(config.d_model, llm_dim)
        self.additional_layer_1 = WhisperEncoderLayer(extended_config)
        # self.additional_layer_2 = WhisperEncoderLayer(extended_config)

        self.ctc_weight = config.ctc_weight
        if config.sub_sample:
            self.subsample_conv1 = nn.Conv1d(
                in_channels=extended_config.d_model,
                out_channels=extended_config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.subsample_conv2 = nn.Conv1d(
                in_channels=extended_config.d_model,
                out_channels=extended_config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        self.lm_head = nn.Linear(extended_config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.final_dropout)

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        orig_len = self.encoder._get_feat_extract_output_lengths(input_lengths)
        if self.config.sub_sample:
            for _ in range(2):
                orig_len = (orig_len + 1) // 2
        return orig_len

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.dim_matching(hidden_states)
        hidden_states = self.additional_layer_1(
            hidden_states, attention_mask, output_attentions=output_attentions, layer_head_mask=None
        )[0]
        # hidden_states = self.additional_layer_2(hidden_states, attention_mask, output_attentions=output_attentions,layer_head_mask=None)[0]

        hidden_states = self.dropout(hidden_states)
        if self.config.sub_sample:
            hidden_states = self.subsample_conv2(self.subsample_conv1(hidden_states.transpose(1, 2))).transpose(1, 2)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones((input_features.size(0), input_features.size(2)), dtype=torch.long)
            )

            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            if labels[0, 0] == self.config.bos_token_id:
                labels = labels[:, 1:]
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.blank_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
