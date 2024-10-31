# pylint: disable=W0246,W0237

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair
from transformers import PreTrainedModel
from transformers.utils import logging

from models.utils import calculate_output_size

"""
Module with streaming ASR components.
"""


logger = logging.get_logger(__name__)


class CausalConv1d(torch.nn.Conv1d):
    """
    Conv1d, for streaming ASR. Zero padding is applied only from 'left_context' side.
    The convolution is causal, i.e. it cannot "peek" into future frames.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input: torch.Tensor, cached_conv: Optional[torch.Tensor] = None):
        """
        args:
            input: input signals, layout (batch, hid_dim, time1)
            cached_conv: left context, it is prepended to input, layout (batch, hid_dim, time2)

        returns:
            (tensor, cached_conv)
        """
        if cached_conv is None:
            input = F.pad(input, (self.__padding, 0))
        else:
            # breakpoint()

            # (batch, n_channels)
            assert input.shape[:2] == cached_conv.shape[:2], (input.shape[:2], cached_conv.shape[:2])
            # (time)
            assert self.__padding == cached_conv.shape[-1], (self.__padding, cached_conv.shape[-1])

            # concatenate with cached_conv (left context):
            input = torch.cat((cached_conv, input), dim=-1)

            # update cached_conv:
            cached_conv = input[..., -self.__padding :].clone()

        return super().forward(input), cached_conv


class CausalConv2d(nn.Conv2d):
    """
    Vanilla Conv2d, subclassed so the name `CausalConv2d` appears in the model.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        if len(padding) != 2:  # padding already is _pair()
            raise ValueError(f"len(padding) must be 2, but we have {len(padding)}")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor):
        return super().forward(input_tensor)


class FeatureExtractorForStreaming(PreTrainedModel):
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = calculate_output_size(
                input_lengths, kernel_size, stride, left_padding=kernel_size - 1 if self.config.is_causal else 0
            )

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = calculate_output_size(
                    input_lengths,
                    1,
                    self.config.adapter_stride,
                    left_padding=self.config.adapter_kernel - 1 if self.config.is_causal else 0,
                )

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
