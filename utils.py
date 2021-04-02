"""utils.py - Helper functions for building the model and for loading model parameters.
   These helper functions are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F


################################################################################
# Help functions for model architecture
################################################################################
# Swish and MemoryEfficientSwish: Two implementations of the method
# round_filters and round_repeats:
#     Functions to calculate params for scaling model width and depth ! ! !
# get_width_and_height_from_size and calculate_output_image_size
# drop_connect: A structural design
# get_same_padding_conv2d:
#     Conv2dStaticSamePadding

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


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


# def get_width_and_height_from_size(x):
#     """Obtain height and width from x.

#     Args:
#         x (int, tuple or list): Data size.

#     Returns:
#         size: A tuple or list (H,W).
#     """
#     if isinstance(x, int):
#         print('*' * 80)
#         return x, x
#     if isinstance(x, list) or isinstance(x, tuple):
#         print('-' * 80)
#         return x
#     else:
#         raise TypeError()


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


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names ! ! !

def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    return partial(Conv2dStaticSamePadding, image_size=image_size)

def my_get_same_padding_conv2d(image_size, weight_size, dilation, stride):
    _kernel_size = [x for x in weight_size[-len(stride):]]
    _pad_size = [max((math.ceil(_i_s/_s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
                    for _i_s, _k_s, _d, _s in zip(image_size, _kernel_size, dilation, stride)]
    _paddings = [(_p//2, _p - _p//2) for _p in _pad_size]

    # unroll list of tuples to tuples
    _paddings = [outer for inner in reversed(
        _paddings) for outer in inner]
    return _paddings

class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(
            image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

        # _stride = [x for x in self.stride]
        # _dilation = [x for x in self.dilation]
        # _input_size = [x for x in image_size]
        _kernel_size = [x for x in self.weight.size()[-len(self.stride):]]
        # _output_size = [math.ceil(_i_s/_s) for _i_s, _s in zip(_input_size, _stride)]
        _pad_size = [max((math.ceil(_i_s/_s) - 1) * _s + (_k_s - 1) * _d + 1 - _i_s, 0)
                     for _i_s, _k_s, _d, _s in zip(image_size, _kernel_size, self.dilation, self.stride)]

        _paddings = [(_p//2, _p - _p//2) for _p in _pad_size]
        # unroll list of tuples to tuples
        self._paddings = [outer for inner in reversed(
            _paddings) for outer in inner]
        val = [int(orig == new) for orig, new in zip([pad_w // 2, pad_w - pad_w // 2,
                                                      pad_h // 2, pad_h - pad_h // 2], self._paddings)]
        assert sum(val) == len(val)

    def forward(self, x):
        # print(x.shape)
        x = self.static_padding(x)
        # print(self.static_padding)
        # print(x.shape)
        # print()
        # print(self.weight.shape)
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x
