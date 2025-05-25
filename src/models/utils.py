import torch
from transformers import PretrainedConfig
from transformers.models.speech_to_text.modeling_speech_to_text import Conv1dSubsampler
from typing import Optional


def calculate_output_size(input_size, kernel_size, stride, left_padding=0, right_padding=0, dilation=1):
    """
    Calculate the output size after a convolution operation with separate left and right padding.

    Parameters:
    - input_size: Input size (width or height)
    - kernel_size: Kernel size
    - stride: Stride
    - left_padding: Left padding
    - right_padding: Right padding

    Returns:
    - Output size
    """
    if not isinstance(input_size, torch.Tensor):
        input_size = torch.tensor(input_size)
    return (((input_size + left_padding + right_padding - dilation * (kernel_size - 1) - 1) / stride) + 1).floor()


def calculate_output_size_multilayer(input_size, layers):
    """
    Calculate the output size after multiple stacked convolution layers.

    Parameters:
    - input_size: Initial input size (width or height)
    - layers: List of tuples (kernel_size, stride, padding) for each layer

    Returns:
    - Final output size after all layers
    """
    current_size = input_size
    for layer in layers:
        kernel_size, stride, left_padding, right_padding = layer
        current_size = calculate_output_size(current_size, kernel_size, stride, left_padding, right_padding)
    return current_size

# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
# added by: Simon Sedlacek
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: Optional[int], decoder_start_token_id: Optional[int]):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id if decoder_start_token_id is not None else 0

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    # print("shifted_input_ids", shifted_input_ids.shape)
    # print("pad_token_id", pad_token_id)
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# added by: Simon Sedlacek
class SpeechEncoderOutputSubsampler(Conv1dSubsampler):
    """This is an extended implementation of the Conv1dSubsampler class from
    speech2text with the addition of the attention mask subsampling methods
    that are normally implemented as methods of the Speech2Text models.
    """

    default_config = PretrainedConfig.from_dict({
            'num_conv_layers': 2,
            'input_feat_per_channel': 256,
            'input_channels': 1,
            'conv_channels': 1024,
            'd_model': 256,
            'conv_kernel_sizes': [5, 5],
        })

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers. Copied from Speech2Text.
        """
        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        """
        Copied from Speech2Text.
        """
        # generate creates 3D attention mask, because of the shape of input_features
        # convert it to 2D if thats the case
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]
        attention_mask = torch.zeros(
            (bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        attention_mask[(torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask

def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float() #(FIX:MZY):return torch.Tensor type
