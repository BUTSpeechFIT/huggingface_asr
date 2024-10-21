""" PyTorch Wav2Vec2-Ebranchformer model."""

from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
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
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerSelfAttention as Wav2Vec2EBranchformerSelfAttention,
)
from transformers.utils import logging

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


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__()

        n_channels = config.intermediate_size // 2  # split input channels
        self.norm = torch.nn.LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            config.csgu_kernel_size,
            1,
            (config.csgu_kernel_size - 1) // 2,
            groups=n_channels,
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


class Wav2Vec2EBranchformerModel(Wav2Vec2ConformerModel):
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
