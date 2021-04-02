"""utils.py - Helper functions for building the model and for loading model parameters.
   These helper functions are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import math
import torch
from torch import nn

padders = [nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]

def round_filters(filters, width_coefficient=None, depth_divisor=None):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    filters *= multiplier

    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient=None):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output

def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = input_image_size
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]

def get_same_padding_conv2d(image_size, weight_size, dilation, stride):
    # weights will always have format [Cout, Cin, *Spatial_dims] <- covers 1D, 2D, 3D .... ND cases
    # therefore number of Spatial_dims can be calculated as len(weight_size)-2
    num_dims = len(weight_size) - 2
    
    # additional checks to populate dilation and stride (in case they are integers or single entry list)
    if isinstance(stride, int):
        stride = (stride,) * num_dims
    elif len(stride) == 1:
        stride = stride * num_dims
    
    if isinstance(dilation, int):
        dilation = (dilation,) * num_dims
    elif len(dilation) == 1:
        dilation = dilation * num_dims

    _kernel_size = [x for x in weight_size[-num_dims:]]
    _pad_size = [max((math.ceil(_i_s/_s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
                    for _i_s, _k_s, _d, _s in zip(image_size, _kernel_size, dilation, stride)]
    _paddings = [(_p//2, _p - _p//2) for _p in _pad_size]

    # unroll list of tuples to tuples
    _paddings = [outer for inner in reversed(
        _paddings) for outer in inner]
    return _paddings

def make_same_padder(conv_op, image_size):
    padding = get_same_padding_conv2d(image_size, conv_op.weight.size(), conv_op.dilation, conv_op.stride)
    if sum(padding) > 0:
        return padders[len(padding)//2-1](padding=padding, value=0)
    else:
        return nn.Identity()
