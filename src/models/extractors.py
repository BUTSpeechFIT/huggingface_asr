from typing import Optional, Union

import torch
from torch import nn
from transformers.activations import ACT2FN

from models.streaming_modules import CausalConv2d
from models.utils import calculate_output_size_multilayer


class Conv2dFeatureExtractor(nn.Module):
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
                    )
                    if hasattr(config, "is_causal") and config.is_causal
                    else nn.Conv2d(
                        conv_in,
                        out_channels=conv_out,
                        kernel_size=(conv_kernel, conv_kernel),
                        stride=(conv_stride, conv_stride),
                        padding=1,
                    ),
                    ACT2FN[config.feat_extract_activation],
                )
                for conv_in, conv_out, conv_kernel, conv_stride in zip(
                    [1, *config.conv_dim], config.conv_dim, config.conv_kernel, config.conv_stride
                )
            ],
        )
        linear_in_dim = config.conv_dim[-1] * int(
            calculate_output_size_multilayer(
                config.second_dim_input_size,
                [
                    (conv_kernel, conv_stride, 1, 0)
                    for conv_kernel, conv_stride in zip(config.conv_kernel, config.conv_stride)
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
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        # pylint: disable=no-member
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # pylint: disable=no-member
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(
                # pylint: disable=no-member
                input_lengths + 2 if isinstance(self.base_model.feature_extractor, Conv2dFeatureExtractor) else 0,
                kernel_size,
                stride,
            )

        if add_adapter:
            # pylint: disable=no-member
            for _ in range(self.config.num_adapter_layers):
                # pylint: disable=no-member
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
