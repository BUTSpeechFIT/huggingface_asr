""" PyTorch Wav2Vec2-Ebranchformer model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForPreTrainingOutput,
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
from vector_quantize_pytorch.random_projection_quantizer import (
    RandomProjectionQuantizer,
)

from models.streaming_modules import CausalConv1d, FeatureExtractorForStreaming

logger = logging.get_logger(__name__)


class Wav2Vec2EBranchformerConfig(Wav2Vec2ConformerConfig, Wav2Vec2Config):
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
        batch_size, sequence_length, hidden_size = hidden_states.size()

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
                config.csgu_kernel_size,
                1,
                (config.csgu_kernel_size - 1) // 2,
                groups=n_channels,
            )
            if config.is_causal
            else torch.nn.Conv1d(
                n_channels,
                n_channels,
                config.csgu_kernel_size,
                1,
                (config.csgu_kernel_size - 1) // 2,
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


class Wav2Vec2EBranchformerModel(FeatureExtractorForStreaming, Wav2Vec2ConformerModel):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.encoder = Wav2Vec2EBranchformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()


class Wav2Vec2EBranchformerForPreTraining(Wav2Vec2ForPreTraining):
    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.post_init()


class Wav2Vec2EBranchformerForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.post_init()


class BestRQEBranchformerConfig(Wav2Vec2EBranchformerConfig):
    model_type = "bestrq-ebranchformer"

    def __init__(
        self,
        best_rq_codebook_size=8192,
        best_rq_codebook_dim=16,
        best_rq_num_books=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.best_rq_codebook_size = best_rq_codebook_size
        self.best_rq_codebook_dim = best_rq_codebook_dim
        self.best_rq_num_books = best_rq_num_books


class BestRQEBranchformerForPreTraining(Wav2Vec2ForPreTraining):
    config_class = BestRQEBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: BestRQEBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        self.post_init()
        self.rpqs = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.conv_dim[-1], elementwise_affine=True),
                RandomProjectionQuantizer(
                    dim=config.conv_dim[-1],
                    codebook_size=config.best_rq_codebook_size,
                    codebook_dim=config.best_rq_codebook_dim,
                    norm=False,
                ),
            )
            for _ in range(config.best_rq_num_books)
        )
        for rpq in self.rpqs:
            rpq.requires_grad = False
        self.classifiers = nn.ModuleList(
            nn.Sequential(nn.Linear(config.hidden_size, config.best_rq_codebook_size), nn.LogSoftmax(dim=-1))
            for _ in range(config.best_rq_num_books)
        )

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        extract_features = outputs[1]
        last_hidden_states = outputs[0]

        loss = None
        for classifier, rpq in zip(self.classifiers, self.rpqs):
            probs = classifier(last_hidden_states)
            labels = rpq(extract_features)
            # pylint: disable=invalid-unary-operand-type
            labels.masked_fill_(~mask_time_indices, -100)

            loss_local = nn.functional.cross_entropy(probs.transpose(1, 2), labels, reduction="sum")
            if loss is None:
                loss = 1 / len(self.rpqs) * loss_local
            else:
                loss += 1 / len(self.rpqs) * loss_local

        if not return_dict:
            if loss is not None:
                return (loss, last_hidden_states, None, None) + outputs[2:]
            return (last_hidden_states, None, None) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=last_hidden_states,
            codevector_perplexity=torch.zeros(1, device=loss.device),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=torch.zeros(1, device=loss.device),
            diversity_loss=torch.zeros(1, device=loss.device),
        )
