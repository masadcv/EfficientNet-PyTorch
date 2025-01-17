"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

from utils import calculate_output_image_size, decode_block_list, drop_connect, make_same_padder, round_filters, round_repeats

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',
)

efficientnet_params = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    }

url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

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

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, 
                        se_ratio, id_skip, batch_norm_momentum, batch_norm_epsilon,
                        image_size=None, drop_connect_rate=0.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.id_skip = id_skip  # whether to use skip connection and drop connect
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self._drop_connect_rate = drop_connect_rate

        # self._block_args = block_args
        bn_mom = 1 - batch_norm_momentum # pytorch's difference from tensorflow
        bn_eps = batch_norm_epsilon
        
        
        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        oup = in_channels * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_conv_padding = make_same_padder(self._expand_conv, image_size)    

            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size, stride=self.stride, bias=False)
        self._depthwise_conv_padding = make_same_padder(self._depthwise_conv, image_size)    
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_reduce_padding = make_same_padder(self._se_reduce, (1, 1))
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._se_expand_padding = make_same_padder(self._se_expand, (1, 1))

        # Pointwise convolution phase
        final_oup = out_channels
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_conv_padding = make_same_padder(self._project_conv, image_size)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=bn_mom, eps=bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(self._expand_conv_padding(inputs))
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(self._depthwise_conv_padding(x))
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(self._se_reduce_padding(x_squeezed))
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(self._se_expand_padding(x_squeezed))
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(self._project_conv_padding(x))
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.in_channels, self.out_channels
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if self._drop_connect_rate:
                x = drop_connect(x, p=self._drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        
        
        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """
    def __init__(self, model_name, blocks_args, in_channels=3, num_classes=1000, 
                width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, 
                image_size=224, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, 
                drop_connect_rate=0.2, depth_divisor=8,):
        super().__init__()
        
        assert model_name in VALID_MODELS, 'model_name should be one of: ' + ', '.join(VALID_MODELS)
        # blocks_args, _ = get_model_params(model_name, {'num_classes': num_classes})
        blocks_args = decode_block_list(blocks_args)
        # print(blocks_args)
        # model = cls(blocks_args, global_params)

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        
        # self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - batch_norm_momentum
        bn_eps = batch_norm_epsilon

        if isinstance(image_size, int):
            image_size = [image_size] * 2

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._conv_stem_padding = make_same_padder(self._conv_stem, image_size)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = []
        num_blocks = 0
        
        # Update block input and output filters based on depth multiplier.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, depth_coefficient)
            )
            self._blocks_args[idx] = block_args
            
            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat
        
        idx = 0
        for block_args in self._blocks_args:
            
            drop_connect_rate = drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / num_blocks # scale drop connect_rate

            # The first block needs to take care of stride and filter size increase.
            # self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size, drop_connect_rate=drop_connect_rate))
            self._blocks.append(MBConvBlock(block_args.input_filters, block_args.output_filters, block_args.kernel_size, 
                                            block_args.stride, block_args.expand_ratio, block_args.se_ratio, block_args.id_skip, 
                                            batch_norm_momentum, batch_norm_epsilon, image_size=image_size, drop_connect_rate=drop_connect_rate))
            idx += 1

            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                drop_connect_rate = drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / num_blocks # scale drop connect_rate
                self._blocks.append(MBConvBlock(block_args.input_filters, block_args.output_filters, block_args.kernel_size, block_args.stride, block_args.expand_ratio, 
                        block_args.se_ratio, block_args.id_skip, batch_norm_momentum, batch_norm_epsilon, 
                        image_size=image_size, drop_connect_rate=drop_connect_rate))
                idx += 1
        self._blocks = nn.Sequential(*self._blocks)
        assert len(self._blocks) == num_blocks

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = make_same_padder(self._conv_head, image_size)    
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(out_channels, num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)
    
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

def from_pretrained(model_name, weights_path=None,
                    in_channels=3, num_classes=1000, **override_params):
    """create an efficientnet model according to name.
    """
    
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]

    wc, dc, isize, dr = efficientnet_params[model_name]
    model = EfficientNet(model_name, blocks_args, in_channels=in_channels, num_classes=num_classes, 
                        width_coefficient=wc, depth_coefficient=dc, dropout_rate=dr, image_size=isize)

    load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000))
    return model

def get_image_size(model_name):
    """Get the input image size for a given efficientnet model.
    """
    # cls._check_model_name_is_valid(model_name)
    assert model_name in VALID_MODELS, 'model_name should be one of: ' + ', '.join(VALID_MODELS)
    _, _, res, _ = efficientnet_params[model_name]
    return res

def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True):
    """Loads pretrained weights from weights path or download using url.

    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ['_fc.weight', '_fc.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    print('Loaded pretrained weights for {}'.format(model_name))