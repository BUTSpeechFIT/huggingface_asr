""" PyTorch Wav2Vec2-Ebranchformer model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _HIDDEN_STATES_START_POSITION,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
    Wav2Vec2GumbelVectorQuantizer,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeedForward as Wav2Vec2EBranchformerFeedForward,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerModel,
    Wav2Vec2ConformerSelfAttention,
)
from transformers.utils import logging

from models.extractors import CustomFE, CustomFEConfig
from models.streaming_modules import CausalConv1d

logger = logging.get_logger(__name__)


class Wav2Vec2EBranchformerConfig(Wav2Vec2ConformerConfig, Wav2Vec2Config, CustomFEConfig):
    """Config for EBranhformer model extending conformer."""

    model_type = "wav2vec2-ebranchformer"

    def __init__(
        self,
        ebranchformer_conv_dropout=0.1,
        csgu_activation="identity",
        csgu_kernel_size=31,
        csgu_use_linear_after_conv=False,
        merge_conv_kernel=31,
        use_macaron_ff=True,
        is_causal=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # EBranchformer related params
        self.csgu_kernel_size = csgu_kernel_size
        self.csgu_activation = csgu_activation
        self.csgu_conv_dropout = ebranchformer_conv_dropout
        self.csgu_use_linear_after_conv = csgu_use_linear_after_conv
        self.merge_conv_kernel = merge_conv_kernel
        self.use_macaron_ff = use_macaron_ff
        self.is_causal = is_causal


class Wav2Vec2EBranchformerSelfAttention(Wav2Vec2ConformerSelfAttention):
    """Self-attention layer for EBranchformer."""

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.is_causal = config.is_causal

    def get_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, _, __ = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.is_causal:
            causal_mask = self.get_causal_mask(query.size(-2), key.size(-2), device=query.device)
            if attention_mask is None:
                attention_mask = causal_mask * -torch.finfo(query.dtype).max
            else:
                attention_mask = attention_mask.masked_fill(causal_mask, -torch.finfo(query.dtype).max)

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores + attention_mask

        # => (batch, head, time1, time2)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # => (batch, head, time1, d_k)
        hidden_states = torch.matmul(probs, value)

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, probs


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__()

        n_channels = config.intermediate_size // 2  # split input channels
        self.norm = torch.nn.LayerNorm(n_channels)
        self.conv = (
            CausalConv1d(
                n_channels,
                n_channels,
                kernel_size=config.csgu_kernel_size,
                stride=1,
                dilation=1,
                groups=n_channels,
            )
            if config.is_causal
            else torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=config.csgu_kernel_size,
                stride=1,
                padding=(config.csgu_kernel_size - 1) // 2,
                dilation=1,
                groups=n_channels,
            )
        )

        if config.csgu_use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if config.csgu_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = ACT2FN[config.csgu_activation]

        self.dropout = torch.nn.Dropout(config.csgu_conv_dropout)

    def forward(self, hidden_states: torch.FloatTensor):
        """Forward method

        Args:
            hidden_states (torch.Tensor): (N, T, D)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = hidden_states.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        hidden_states = x_r * x_g  # (N, T, D/2)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__()
        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.intermediate_size), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(config)
        self.channel_proj2 = torch.nn.Linear(config.intermediate_size // 2, config.hidden_size)

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.channel_proj1(hidden_states)  # hidden_size -> intermediate_size
        hidden_states = self.csgu(hidden_states)  # intermediate_size -> intermediate_size/2
        hidden_states = self.channel_proj2(hidden_states)  # intermediate_size/2 -> hidden_size
        return hidden_states


class Wav2Vec2EBranchformerEncoderLayer(nn.Module):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        if config.use_macaron_ff:
            self.ff1 = nn.Sequential(nn.LayerNorm(embed_dim), Wav2Vec2EBranchformerFeedForward(config))

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.self_attn = Wav2Vec2EBranchformerSelfAttention(config)

        # cgMLP
        self.cgMLP = ConvolutionalGatingMLP(config)
        self.cgMLP_layer_norm = nn.LayerNorm(config.hidden_size)
        self.cgMLP_dropout = torch.nn.Dropout(dropout)

        # Merge
        self.final_dropout = torch.nn.Dropout(dropout)
        self.merge_proj = torch.nn.Linear(embed_dim + embed_dim, embed_dim)
        self.depthwise_conv_fusion = torch.nn.Conv1d(
            embed_dim + embed_dim,
            embed_dim + embed_dim,
            kernel_size=config.merge_conv_kernel,
            stride=1,
            padding=(config.merge_conv_kernel - 1) // 2,
            groups=embed_dim + embed_dim,
            bias=True,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # Feed-forward 2
        if config.use_macaron_ff:
            self.ff2 = nn.Sequential(nn.LayerNorm(embed_dim), Wav2Vec2EBranchformerFeedForward(config))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 1. Optional ff1
        if self.ff1:
            residual = hidden_states
            hidden_states = residual + 0.5 * self.ff1(hidden_states)

        # 2. Split input to three branches
        residual = hidden_states
        global_branch = hidden_states
        local_branch = hidden_states

        # 3. Self-Attention branch
        global_branch = self.self_attn_layer_norm(global_branch)
        global_branch, attn_weigts = self.self_attn(
            hidden_states=global_branch,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        global_branch = self.self_attn_dropout(global_branch)

        # 4. cgMLP Branch
        local_branch = self.cgMLP_layer_norm(local_branch)
        local_branch = self.cgMLP(local_branch)

        # 5. Merge operator
        # a, concat
        hidden_states = torch.cat([global_branch, local_branch], dim=-1)
        merge_residual = hidden_states
        # b, depth-wise conv mixing
        hidden_states = merge_residual + self.depthwise_conv_fusion(hidden_states.transpose(1, 2)).transpose(1, 2)
        # c, project back to original size and final dropout
        hidden_states = self.final_dropout(self.merge_proj(hidden_states))

        # 6. Add residual
        hidden_states = residual + hidden_states

        # 7. Optional ff2
        if self.ff2:
            residual = hidden_states
            hidden_states = residual + 0.5 * self.ff2(hidden_states)

        # 8. Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, attn_weigts


class Wav2Vec2EBranchformerEncoder(Wav2Vec2ConformerEncoder):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Wav2Vec2EBranchformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.pos_conv_embed = None


class Wav2Vec2EBranchformerModel(CustomFE, Wav2Vec2ConformerModel):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        Wav2Vec2ConformerModel.__init__(self, config)

        self.encoder = Wav2Vec2EBranchformerEncoder(config)

        self.overwrite_fe(config)

        # Initialize weights and apply final processing
        self.post_init()


class Wav2Vec2GumbelVectorQuantizerCustom(Wav2Vec2GumbelVectorQuantizer):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        super().__init__(config)
        self.weight_proj = nn.Linear(config.hidden_size, self.num_groups * self.num_vars)


class Wav2Vec2EBranchformerForPreTraining(Wav2Vec2ForPreTraining):
    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.quantizer = Wav2Vec2GumbelVectorQuantizerCustom(config)
        if hasattr(self.wav2vec2, "masked_spec_embed"):
            del self.wav2vec2.masked_spec_embed
        self.post_init()


#
#
# class Wav2Vec2EBranchformerEncoderLayerWrapper(Wav2Vec2EBranchformerEncoderLayer):
#     def __init__(self, config: Wav2Vec2EBranchformerConfig, wav2vec2: Wav2Vec2EBranchformerModel):
#         super().__init__(config)
#         self.wav2vec2 = wav2vec2
#
#     def forward(
#             self,
#             hidden_states: torch.FloatTensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             relative_position_embeddings: Optional[torch.Tensor] = None,
#             output_attentions: bool = False,
#             input_values: Optional[torch.Tensor] = None,
#     ):
#
#         # if self.wav2vec2.encoder.embed_positions is not None:
#         #     relative_position_embeddings = self.wav2vec2.encoder.embed_positions(hidden_states)
#         # else:
#         #     relative_position_embeddings = None
#         extract_features = self.wav2vec2.feature_extractor(input_values)
#         extract_features = extract_features.transpose(1, 2)
#         if attention_mask is not None:
#             # compute reduced attention_mask corresponding to feature vectors
#             attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
#                 extract_features.shape[1], attention_mask, add_adapter=False
#             )
#         if attention_mask is not None:
#             # make sure padded tokens output 0
#             hidden_states[~attention_mask] = 0.0
#
#             # extend attention_mask
#             attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
#             attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
#             attention_mask = attention_mask.expand(
#                 attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
#             )
#
#         return super().forward(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             relative_position_embeddings=relative_position_embeddings,
#             output_attentions=output_attentions,
#         )
#


class Wav2Vec2EBranchformerForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.blank_projection = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def freeze_encoder(self):
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = self.dropout(hidden_states)

        if output_hidden_states is not True:
            outputs.hidden_states = None

        logits = self.lm_head(hidden_states)
        logits = torch.concatenate((logits, self.blank_projection(hidden_states)), dim=-1)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
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
                    blank=logits.shape[-1] - 1,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
