"""This module contains the feature extractors for the ASR model."""
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from models.streaming_modules import CausalConv2d
from models.utils import calculate_output_size_multilayer


logger = logging.get_logger(__name__)

class CustomFEConfig(PretrainedConfig):
    """This class contains the configuration for the feature extractor."""

    def __init__(self, conv_padding=(1, 1), num_fbanks=80, context_awareness_type=None, **kwargs):
        super().__init__(**kwargs)
        self.conv_padding = conv_padding
        self.num_fbanks = num_fbanks
        self.context_awareness_type = context_awareness_type


class GatedConv2d(nn.Module):
    """This class implements the gated convolutional layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states) * torch.sigmoid(self.gate(hidden_states))


class GatedConv2dShared(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, shared_scale_factor=4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.shared_scale_factor = shared_scale_factor
        self.gate = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size[0] * self.shared_scale_factor, kernel_size[1]),
            stride=(stride[0] * self.shared_scale_factor, stride[1]),
            padding=(padding * self.shared_scale_factor, padding),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(hidden_states)
        gate_out = torch.sigmoid(self.gate(hidden_states))
        conv_out_reshaped = conv_out.view(*conv_out.size()[:2], -1, self.shared_scale_factor, conv_out.size(3))
        out = (conv_out_reshaped * gate_out.unsqueeze(3)).view(conv_out.size())
        return out


class ContextAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, context_awareness_type=None):
        super().__init__()
        context_awareness_module_mapping = {"gated": GatedConv2d, "gated_shared": GatedConv2dShared}
        conv_class = context_awareness_module_mapping.get(context_awareness_type, nn.Conv2d)
        self.conv = conv_class(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class Conv2dFeatureExtractor(nn.Module):
    """
    config.conv_padding:
        padding for Conv2d, pads the input on all 4 sides.

        option1:
        `[[ time_pad1, dim_pad1], [ time_pad2, dim_pad2 ]]`
        pads separately rows and columns in the 2 Conv2d modules.
        `time_pad1` adds frames both to the beginning and the
        end of the sequence.

        option2:
        `[ pad1, pad2 ]`
        same amount of padding from all 4 sides (top, down, left, right).
        `pad1` for 1st module, `pad2` for 2nd module.
    """
    def __init__(self, config):
        super().__init__()

        self.conv = torch.nn.Sequential(
            *[
                nn.Sequential(
                    CausalConv2d(
                        conv_in,
                        out_channels=conv_out,
                        kernel_size=(conv_kernel, conv_kernel),
                        stride=(conv_stride, conv_stride),
                        padding=_pair(conv_padding),
                    )
                    if hasattr(config, "is_causal") and config.is_causal
                    else ContextAwareConv2d(
                        conv_in,
                        out_channels=conv_out,
                        kernel_size=(conv_kernel, conv_kernel),
                        stride=(conv_stride, conv_stride),
                        padding=_pair(conv_padding),
                        context_awareness_type=config.context_awareness_type,
                    ),
                    ACT2FN[config.feat_extract_activation],
                )
                for conv_in, conv_out, conv_kernel, conv_stride, conv_padding in zip(
                    [1, *config.conv_dim], config.conv_dim, config.conv_kernel, config.conv_stride, config.conv_padding
                )
            ],
        )

        linear_in_dim = config.conv_dim[-1] * int(
            calculate_output_size_multilayer(
                config.num_fbanks,
                [
                    (
                        conv_kernel,
                        conv_stride,
                        _pair(conv_padding)[1],  # left_padding (not on time axis)
                        _pair(conv_padding)[1],  # right_padding (not on time axis)
                    )
                    for conv_kernel, conv_stride, conv_padding in zip(
                        config.conv_kernel, config.conv_stride, config.conv_padding,
                    )
                ],
            )
        )
        self.out = torch.nn.Linear(linear_in_dim, config.hidden_size, bias=True)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(input_values[:, None, ...])
        hidden_states = self.out(hidden_states.transpose(1, 2).flatten(2, 3))
        return hidden_states.transpose(1, 2)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


class CustomFE:
    # pylint: disable=no-member
    def __init__(self):
        self.config = None
        self.feature_projection = None
        self.feature_extractor = None

    def overwrite_fe(self, config):
        self.config = config
        self.feature_extractor = Conv2dFeatureExtractor(config)
        self.feature_projection.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.feature_projection.projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        # pylint: disable=no-member
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride, time_padding):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            #
            # assumed: dilation=1
            return torch.div(input_length + time_padding - kernel_size, stride, rounding_mode="floor") + 1

        # pylint: disable=no-member
        for kernel_size, stride, padding in zip(
            self.config.conv_kernel, self.config.conv_stride, self.config.conv_padding
        ):
            input_lengths = _conv_out_length(
                # pylint: disable=no-member
                input_lengths,
                kernel_size,
                stride,
                2 * _pair(padding)[0],  # top_padding + bottom_padding (on time axis)
            )

        if add_adapter:
            raise NotImplementedError("Adapter layers are not implemented yet.")

        return input_lengths
