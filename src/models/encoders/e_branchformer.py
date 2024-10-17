""" PyTorch Wav2Vec2-Ebranchformer model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    Wav2Vec2BaseModelOutput,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        cached_key: Optional[torch.Tensor] = None,
        cached_value: Optional[torch.Tensor] = None,
        left_context_len: int = 0,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        both streaming + non-streaming version

        for streaming, call with: (cached_key, cached_value, left_context_len)
        """

        # self-attention mechanism
        batch_size, num_frames, hid_dim = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # still relevant ?
        if self.position_embeddings_type == "rotary":
            raise ValueError("position_embeddings_type == 'rotary' not supported")

        # project query_key_states and value_states => (batch, time1, head, d_k)
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # prepend the context of 'key' matrix
        if cached_key != None:
            assert(cached_key.shape[2] == left_context_len)
            key = torch.cat([cached_key, key], dim=2)

            # update the cached_key
            if left_context_len > 0:
                cached_key = key[..., -left_context_len:, :]
            elif left_context_len == 0:
                shape = list(key.shape)
                shape[-2] = 0
                cached_key = torch.zeros(shape, device=key.device)

        # prepend the context of 'value' matrix
        if cached_value != None:
            assert(cached_value.shape[2] == left_context_len)
            value = torch.cat([cached_value, value], dim=2)

            # update the cached_key
            if left_context_len > 0:
                cached_value = value[..., -left_context_len:, :]
            elif left_context_len == 0:
                shape = list(value.shape)
                shape[-2] = 0
                cached_value = torch.zeros(shape, device=key.device)

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

        # apply attention_mask if necessary (prepared in Encoder class)
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

        return hidden_states, probs, (cached_key, cached_value,)


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
        cached_key: Optional[torch.Tensor] = None,
        cached_value: Optional[torch.Tensor] = None,
        left_context_len: int = 0,
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
        global_branch, attn_weigts, (cached_key, cached_value) = self.self_attn.forward(
            hidden_states=global_branch,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            cached_key=cached_key,
            cached_value=cached_value,
            left_context_len=left_context_len,
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
        return hidden_states, attn_weigts, (cached_key, cached_value)


class Wav2Vec2EBranchformerEncoder(Wav2Vec2ConformerEncoder):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Wav2Vec2EBranchformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.pos_conv_embed = None

    def get_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def build_attention_mask(
        self,
        hidden_states: Tensor,
        attention_lens: Optional[Tensor] = None,
        chunk_size: int = -1,
        left_context_len: int = 0,
        is_streaming_inference: bool = False,
    ) -> Tensor:
        """
        build attention mask

        attention_lens: 2D tensor with shape (batch_size, embed_time),
                        1.0 for segment frames, 0.0 for padding after segment

        we use 3 types:
        - streaming inference: unlimited access of SelfAttention
        - training without chunks: causal mask
        - training with chunks: block-diagonal mask (small look-ahead, left context)
        """

        if attention_lens is None:
            logger.error("Missing `attention_lens`, cannot create `attention_mask` for SelfAttention module.")
            return None

        assert chunk_size == -1 or chunk_size > 0, chunk_size
        assert left_context_len >= 0, left_context_len

        if is_streaming_inference:
            # streaming_decode mask -> unlimited access within the chunk's length (no causal masking)
            time1 = attention_mask.shape[-1]
            time2 = time1 + left_context_len

            # extend attention_lens for `left_context_len`
            left_attention_lens = torch.ones(
                attention_lens.shape[0], left_context_len, dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_lens = torch.cat([left_attention_lens, attention_lens], dim=1)

            # expand the attention_mask to "attention-prob" shape
            attention_mask = 1.0 - attention_lens[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

            # Note: allowing attention look-ahead within the current chunk (no causal mask used)
            return attention_mask

        if chunk_size == -1:
            # training mask, no chunking -> length masking & causal masking
            attention_mask = 1.0 - attention_lens[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

            if self.is_causal:
                causal_mask = self.get_causal_mask(
                    attention_mask.shape[-1], attention_mask.shape[-1], device=query.attention_mask,
                )
                attention_mask = torch.logical_or(attention_mask, causal_mask)

            # set the negative value
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            return attention_mask

        else:
            # training mask, chunk_size set -> length masking & block-diaglonal mask
            attention_mask = 1.0 - attention_lens[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

            # block diagonal mask
            num_chunks = math.ceil(attention_mask.shape[-1] / chunk_size)
            block_diagonal_mask = torch.ones(
                attention_mask.shape[-1], attention_mask.shape[-1], device=attention_mask.device,
            )
            for i in range(num_chunks):
                block_diagonal_mask[
                    i*chunk_size : (i+1)*chunk_size,
                    max(i*chunk_size - left_context_len, 0) : (i+1)*chunk_size
                ] = 0.0

            # superpose the masks
            attention_mask = torch.logical_or(attention_mask, block_diagonal_mask)

            # mask-out lines after end of utterance
            lens = attention_lens.sum(dim=1)
            for ii, len_ii in enumerate(lens):
                attention_mask[ii,:, len_ii:,:] = 1.0

            # set the negative value
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            return attention_mask

    def forward(
        self,
        hidden_states: Tensor,
        attention_lens: Optional[Tensor] = None,
        chunk_size: int = -1,
        left_context_len: int = 0,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        copied from transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
        class Wav2Vec2ConformerEncoder

        modified to support block diagonal attention with left context
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        assert chunk_size == -1 or chunk_size > 0, chunk_size
        assert left_context_len >= 0, left_context_len

        # make sure padded tokens output 0
        hidden_states[~attention_lens] = 0.0

        attention_mask = self.build_attention_mask(
            hidden_states=hidden_states,
            attention_lens=attention_lens,
            chunk_size=chunk_size,
            left_context_len=left_context_len,
            is_streaming_inference=False,
        )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        left_context_frames: int = 64,  # embedding time
        device: torch.device = torch.device("cpu"),
    ) -> list[Tensor]:
        """
        Get initial streaming states.
        A list of cached tensors of all encoder layers. For layer-i states[i*2:(i+1)*2]
        is (cached_key, cached_value).
        """
        # Create only the state of the encoder, `processed_lens` are added in
        # `ebranchformer/streaming_decode.py:get_init_states()`.

        streaming_states = []

        num_layers = self.config.num_hidden_layers
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads

        for layer in range(num_layers):
            # layout: (batch, head, time1, d_k)
            cached_key = torch.zeros(
                (batch_size, num_attention_heads, left_context_frames, head_dim),
                device=device,
            )
            cached_value = torch.zeros(
                (batch_size, num_attention_heads, left_context_frames, head_dim),
                device=device,
            )
            streaming_states += [
                cached_key,
                cached_value,
            ]

        return streaming_states

    def streaming_forward(
        self,
        hidden_states: Tensor,
        attention_lens: Optional[Tensor],
        streaming_states: list[Tensor],
        left_context_len: int = 64,
        output_attentions: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Encoder forward function through for streaming ASR.

        hidden_states: pre-encoder outputs, layout (batch, time, dim_encoder).
        attention_mask: chunk lengths encoded as binary matrix with shape (batch, time).
                        Related to SelfAttention time masking.
        streaming_states: vector of stacked states (self-attention keys and values, 1+1 per layer).
        left_context_len: length of the left_context in the SelfAttention for streaming
                          (unit is encoder timestep 40ms).
        output_attentions: whether to export attention matrices from the SelfAttention modules.

        Returns:
            tuple(hidden_states, list(streaming_states), list(attention_out))

        """

        assert len(streaming_states) == 2*len(self.layers), \
                (len(streaming_states), 2*len(self.layers))
        assert attention_out is not None

        new_streaming_states = []
        attention_out = []

        # make sure padded tokens output 0
        if attention_lens is not None:
            hidden_states[~attention_lens] = 0.0

        attention_mask = self.build_attention_mask(
            hidden_states=hidden_states,
            attention_lens=attention_lens,
            is_streaming_inference=True,
        )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            batch_size, num_frames, hid_dim = hidden_states.size()
            device = hidden_states.device
            dtype = hidden_states.dtype

            # Create empty Tensor `ext_key_shape`, corrseponds to shape of `torch.cat([cached_key, key])`
            # in Wav2Vec2EBranchformerSelfAttention, so the dims of `relative_position_embeddings` match...
            left_context_len = streaming_states[0].shape[2]  # length of `cached_key`
            ext_key_shape = torch.zeros((batch_size, num_frames+left_context_len, hid_dim), dtype=dtype, device=device)

            relative_position_embeddings = self.embed_positions(ext_key_shape)
        else:
            relative_position_embeddings = None

        for i, layer in enumerate(self.layers):
            # get streaming state
            cached_key, cached_value = streaming_states[2*i : 2*(i+1)]

            # streaming_forward()
            layer_outputs = layer.streaming_forward(
                hidden_states=hidden_states,
                cached_key=cached_key,
                cached_value=cached_value,
                left_context_len=left_context_len,
                key_padding_mask=None,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
                output_attentions=output_attentions,
            )
            hidden_states, attn_weights, (cached_key, cached_value) = layer_outputs

            # collect new states
            new_streaming_states += [ cached_key, cached_value ]

            # collect attention matrices
            if output_attentions:
                attention_out.append(attn_weights)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, new_streaming_states, attention_out


class Wav2Vec2EBranchformerModel(CustomFE, Wav2Vec2ConformerModel):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        Wav2Vec2ConformerModel.__init__(self, config)

        self.encoder = Wav2Vec2EBranchformerEncoder(config)

        self.overwrite_fe(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        chunk_size: int = -1,
        left_context_len: int = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        """
        Copied from transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py,
        class Wav2Vec2ConformerModel

        extended with `chunk_size` and `left_context_len`
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # convert (chunk_size,left_context) length: fbank -> embedding time
        chunk_size = self._get_feat_extract_output_lengths(
            input_lengths=chunk_size,
            add_adapter=False,
        )
        left_context_len = self._get_feat_extract_output_lengths(
            input_lengths=left_context_len,
            add_adapter=False,
        )

        encoder_outputs = self.encoder.forward(
            hidden_states,
            attention_lens=attention_mask,
            chunk_size=chunk_size,
            left_context_len=left_context_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def streaming_forward(
        self,
        input_values: torch.FloatTensor,
        streaming_states: list[torch.Tensor],
        left_context_len: int,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:

        """
        Forward function for streaming ASR.

        input_values: fbank features, layout (batch, dim_fea, time).
        streaming_states: vector of stacked states (self-attention keys and values, 1+1 per layer).
        left_context_len: length of the left_context in the SelfAttention for streaming (unit is 10ms).
        mask_time_indices: currently unused from outside, related to Spec-augment masking.
        attention_mask: chunk lengths encoded as binary matrix with shape (batch, time).
                        Related to SelfAttention time masking.
        output_attentions: whether to export attention matrices from the SelfAttention modules.

        Returns:
            tuple(hidden_states, list(streaming_states), list(attention_out))

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        extract_features = self.feature_extractor(input_values)
        # (batch, dim_fea, time) -> (batch, time, dim_fea)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            # (convert: 'input' 10ms steps -> 'feature' 40ms steps)
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # apply feature_projection
        hidden_states, extract_features = self.feature_projection(extract_features)
        # apply Spec-augment
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # convert left_context length: fbank -> embedding time
        left_context_len = self._get_feat_extract_output_lengths(
            input_lengths=left_context_len,
            add_adapter=False,
        )

        hidden_states, new_streaming_states, attention_out = self.encoder.streaming_forward(
            hidden_states=hidden_states,
            attention_lens=attention_mask,
            streaming_states=streaming_states,
            left_context_len=left_context_len,
            output_attentions=output_attentions,
        )

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        return hidden_states, new_streaming_states, attention_out


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


class Wav2Vec2EBranchformerForCTC(Wav2Vec2ForCTC):
    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2EBranchformerModel(config)
        if hasattr(self.wav2vec2, "masked_spec_embed"):
            del self.wav2vec2.masked_spec_embed
        self.post_init()
