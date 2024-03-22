import torch


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
    return torch.tensor(
        ((input_size + left_padding + right_padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    ).floor()


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
