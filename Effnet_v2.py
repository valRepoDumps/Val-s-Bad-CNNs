#Implementing effnet, again!
#Implementing effnet, again!
import torch
import math
from torch import nn
from typing import Tuple
from collections import namedtuple
import torch.nn.functional as F

GlobalParams = namedtuple('GlobalParams', #name of the global params
                          [
                              'width_coefficient',
                              'depth_coefficient',
                              'dropout_rate',
                              'drop_connect_rate',
                              'image_size',
                              'color_channels',
                              'num_classes',
                              'batch_norm_momentum',
                              'batch_norm_epsilon',
                              'depth_divisor',
                              #'min_depth', no min depth. me lazy
                              #no include top. im not smar.
                          ])

BlockParams = namedtuple('LayerParams', [
    'num_repeat',
    'kernel_size',
    'stride',
    'expand_ratio',
    'se_reduction_ratio',
    'id_skip',
    'input_channels',
    'output_channels',
])

#set default of these to None
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockParams.__new__.__defaults__ = (None,) * len(BlockParams._fields)


def set_default_params(color_channels, 
                       width_coeff, 
                       depth_coeff, 
                       image_size = None,
                       num_classes = 10, 
                       dropout_rate = 0.1, 
                       drop_connect_rate = 0.1, 
                       batch_norm_momentum = 0.99, 
                       batch_norm_eps = 0.001, 
                       depth_divisor = 8, 
                       args_dict = None):
    """
    A function to generate the global parameters and the blocks parameters for efficientnet. 
    Args:
    color_channels (int): the number of color channels in the image.
    width_coeff: the width coefficient for the model. Used to calculate input and output_channels of blocks
    depth_coeff: the depth coefficient for the model. Used to calculate layers per block
    image_size (int): the size of the image. (defaults to None) #not implemented!
    num_classes: the number of classes for the model. (defaults to 10)
    dropout_rate: the dropout rate for the model. (defaults to 0.1)
    drop_connect_rate: the drop connect rate for the model. (defaults to 0.1)
    batch_norm_momentum: the momentum for the batch normalization. #not implemented!
    batch_norm_eps: the epsilon for the batch normalization. #not implemented!
    depth_divisor: the divisor for the depth of the model. (defaults to 8)
    args_dict: the dictionary of the arguments for the model. (is default if None.)
    """

    if args_dict == None:
        args_dict = [ #num_repeat, kernel_size, stride,  expand_ratio, se_reduction_ratio, id_skip, input_channels, output_channels
            [1, 3, [1,1],  1, 0.25, True,32,16], #3
            [2, 3, [2,2],  6, 0.25, True,16,24], #3
            [2, 5, [2,2],  6, 0.25, True,24,40], #5
            [3, 3, [2,2],  6, 0.25, True,40,80], #3
            [3, 5, [1,1],  6, 0.25, True,80,112], #5
            [4, 5, [2,2],  6, 0.25, True,112,192], #5
            [1, 3, [1,1],  6, 0.25, True,192,320], #3
        ] #default . can be changed with width coeff, and deth coeff so that my colab can acutally run it. 

    blocks_params = []
    for params in args_dict:
        blocks_params.append(BlockParams(*params))

    global_params = GlobalParams(
        width_coefficient = width_coeff,
        depth_coefficient = depth_coeff,
        dropout_rate = dropout_rate,
        drop_connect_rate = drop_connect_rate,
        image_size = None,
        color_channels = color_channels,
        num_classes = num_classes,
        batch_norm_momentum = batch_norm_momentum,
        batch_norm_epsilon = batch_norm_eps,
        depth_divisor = depth_divisor,
    )

    return global_params, blocks_params

def drop_connect(x, drop_ratio: float, training: bool = False): #implement drop connect
    """
    A function to implement drop connect.
    Args:
    x: the input tensor.
    drop_ratio: the ratio of the tensor to be dropped.
    training: whether the model is in training mode.
    """
    assert 0<= drop_ratio <= 1, 'drop_ratio must be in range [0, 1]'

    if training:
        return x

    keep_ratio = 1.0 - drop_ratio

    batch_size = x.shape[0]

    random_tensor = keep_ratio + torch.rand([batch_size, 1, 1, 1], dtype = x.dtype, device = x.device)

    binary_tensor = torch.floor(random_tensor)

    return (x / keep_ratio) * binary_tensor

def get_next_in_channels(current_channels, divisor, multiplier):
    """
    A function to get the next number of channels, knowing the multiplier and divisor. 
    Args:
    current_channels: the current number of channels.
    divisor: the divisor to use. The next channel will be divisible by that divisor.
    multiplier: the multiplier to use. The current channel will be multiplied by that multiplier before being processed.
    """

    current_channels *= multiplier
    new_channels = max(divisor, int(current_channels + divisor / 2) // divisor * divisor) #find the largest number that can divide by the divisor that's around the current channels after multiplication
    if new_channels < 0.9 * current_channels: #avoid rounding down more than 10%
        new_channels += divisor
    return new_channels

def depth_calc(current_depth, depth_coeff):
    """
    A function to calculate the depth of the block.
    Args:
    current_depth: the current depth of the model.
    depth_coeff: the depth coefficient to use.
    """
    if not depth_coeff:
        return current_depth
    return int(math.ceil(depth_coeff * current_depth))


class CustomConv2d(nn.Conv2d):
    """
    A custom convolution layer designed to work with efficientnet.
    Ensures the image size either change drastically (half, //3, //4), or not change at all.
    Args:
    in_channels: the number of input channels.
    out_channels: the number of output channels.
    kernel_size: the size of the kernel.
    stride: the stride of the convolution.
    dilation: the dilation of the convolution.
    groups: the number of groups to use.
    bias: whether to use a bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, dilation = 1, groups = 1, bias = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias) #there's a 0 here. the 0 is the padding. its additional. remove it beark everythiung. 
        
        # self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2 #no idea why theere's conditionals here. infact, ill remove it.

    def forward(self, x):
        ih, iw = x.size()[-2:] #get image height and width
        kh, kw = self.weight.size()[-2:] #get kernel height and width from self.weight.size()
        sh, sw = self.stride #get stride height and width
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0) #do some magic 
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0) #more magic!
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]) #padding using magic numbers we got previously
        
        return F.conv2d(x, weight = self.weight, bias = self.bias, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)

# class CustomConv2d(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, dilation = 1, groups = 1, bias = True):



class NormAct(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            nn.BatchNorm2d(num_features=channels),
            nn.SiLU(),
        )

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernal_size = 3, stride = 1, groups = 1):
        super().__init__(
            NormAct(channels=in_channels),
            CustomConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size, stride=stride, groups=groups)
        )

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduced_dims: int):
        super().__init__()

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels= reduced_dims, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=reduced_dims, out_channels= in_channels, kernel_size= 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se_block(x)
    
class MBConvBlock(nn.Module):
    def __init__(self, block_params, global_params):
        super().__init__()
        
        hidden_units = int(block_params.input_channels * block_params.expand_ratio)
        reduced_dims = int(block_params.input_channels * block_params.se_reduction_ratio)
        
        input_channels = block_params.input_channels
        output_channels = block_params.output_channels
        stride = block_params.stride

        self.MBlayers = nn.Sequential(
            ConvBlock(in_channels=input_channels, out_channels=hidden_units, kernal_size=1),
            ConvBlock(in_channels=hidden_units, out_channels=hidden_units, kernal_size=block_params.kernel_size, stride = stride, groups=hidden_units),
            NormAct(channels=hidden_units),
            SqueezeExcitationBlock(in_channels=hidden_units, reduced_dims = reduced_dims), 
            CustomConv2d(in_channels=hidden_units, out_channels=output_channels, kernel_size=1)
        )

        self.shorcut =self.get_shortcut(in_channels = input_channels, out_channels=output_channels, stride = stride)

        self.gamma = nn.Parameter(torch.zeros(1))   

    def get_shortcut(self, in_channels, out_channels, stride): #make it so x can add to the final output, having residual. 
        
        if isinstance(stride, Tuple) or isinstance(stride, list):
            stride = stride[0]

        layer = nn.Identity() #identity layer does nothing
        
        if stride > 1:
            layer = nn.AvgPool2d(stride)
        
        if in_channels != out_channels:
            layer = nn.Sequential(
                layer,
                CustomConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            )

        return layer
    
    def forward(self, x):
        return self.shorcut(x) + self.MBlayers(x) * self.gamma
    
class BlockStacks(nn.Sequential):
    def __init__(self, blocks_params, global_params):

        layers = []
        
        for block_params in blocks_params:

            in_channels = get_next_in_channels(block_params.input_channels, multiplier = global_params.width_coefficient, divisor = global_params.depth_divisor)
            out_channels = get_next_in_channels(block_params.output_channels, multiplier= global_params.width_coefficient, divisor = global_params.depth_divisor)
            num_repeat = depth_calc(block_params.num_repeat, global_params.depth_coefficient)

            block_params = block_params._replace(
                input_channels = in_channels,
                output_channels = out_channels,
                num_repeat = num_repeat,
            )         #._replace() return a new named tuple, doenst modify existing ones. so = 
            #print(f"Input: {block_params.input_channels} | Output: {block_params.output_channels}")
            #assign block params to the modified channels. 
            
            for _ in range(block_params.num_repeat):
                #print(f"Input: {block_params.input_channels} | Output: {block_params.output_channels}")
                layers.append(
                    MBConvBlock(
                        block_params = block_params,
                        global_params = global_params,
                    ))
                
                block_params = block_params._replace(
                    input_channels = out_channels,
                    stride = [1,1]
                )
        
        super().__init__(*layers)

class Stem(nn.Sequential):
    def __init__(self, color_channels, stride, out_channels, global_params):
        out_channels = get_next_in_channels(out_channels, multiplier = global_params.width_coefficient, divisor = global_params.depth_divisor)
        super().__init__(
            CustomConv2d(in_channels=color_channels, out_channels=out_channels, stride = stride, kernel_size=3),
            NormAct(channels=out_channels)
        )

class Head(nn.Sequential):
    def __init__(self, input_channels, global_params):
        hidden_units = get_next_in_channels(1280, multiplier = global_params.width_coefficient, divisor = global_params.depth_divisor)
        super().__init__(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size= 1),
            NormAct(channels=hidden_units),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p = global_params.dropout_rate),
            nn.Linear(in_features=hidden_units, out_features= global_params.num_classes)
        )

class EfficientNet(nn.Sequential):
    def __init__(self, global_params, blocks_params, color_channels = 3):
        super().__init__(
            Stem(color_channels=global_params.color_channels, stride=1, out_channels=blocks_params[0].input_channels, global_params=global_params),
            BlockStacks(blocks_params=blocks_params, global_params=global_params),
            Head(input_channels=get_next_in_channels(blocks_params[-1].output_channels, multiplier = global_params.width_coefficient, divisor = global_params.depth_divisor),
                 global_params=global_params
                 )
        )


# args_dict = [ #num_repeat, kernel_size, stride,  expand_ratio, se_reduction_ratio, id_skip, input_channels, output_channels
#         [1, 3, [1,1],  1, 0.25, True,32,16], #3
#         [2, 3, [2,2],  6, 0.25, True,16,32], #3
#         [2, 5, [2,2],  6, 0.25, True,32,64], #5
#         [3, 3, [2,2],  6, 0.25, True,64,128], #3
#         [3, 5, [1,1],  6, 0.25, True,128, 256], #5
#         # [4, 5, [2,2],  6, 0.25, True,112,192], #5
#         # [1, 3, [1,1],  6, 0.25, True,192,320], #3
#     ]

# global_params, blocks_params = set_default_params(color_channels = 3, width_coeff =0.25, depth_coeff = 0.5, depth_divisor = 8, dropout_rate = 0.2, args_dict = args_dict)
# model =EfficientNet(global_params=global_params, blocks_params = blocks_params, color_channels=3)


# from utils import get_model_summary
# print(get_model_summary(model=model, input_size=(32,3,32,32)))

# preds = model(torch.rand(32,3,32,32))
