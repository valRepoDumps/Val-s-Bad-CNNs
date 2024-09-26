"""
Contain the Pytorch code to instantiate a models i made. 
"""
import torch
import math
from torch import nn
from typing import List, Tuple, Dict


class DummyModel(nn.Module):
    """
    A class to create a dummy neural network that's easy to train.
    Args:
    in_channels: number of color channels in the image.
    targets: number of output classes.
    image_size: size of the image. (A tuple of HxW or an int)
    """
    def __init__(self, in_channels, targets, image_size: List[int] | int):
        super().__init__()
        if isinstance(image_size, Tuple):
            image_height, image_width = image_size
        elif isinstance(image_size, int):
            image_height, image_width = image_size, image_size
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=image_width*image_height*in_channels, out_features= targets)
        )
    def forward(self, x):
        return self.layers(x)


class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.
    Recreate the TInyVGG architecture from the CNN explainer website.
    Args:
    input_channels (int): the number of input (color) channels.
    hidden_units (int): the number of hidden units.
    output_shape (int): the number of output classes.
    """

    def __init__(self, input_channels: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            #the first convolution layers
            nn.Conv2d(in_channels = input_channels, out_channels = hidden_units, kernel_size = 3, padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #the second convolution layers
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, padding = 0),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #flatten layer
            nn.Flatten(),
            nn.Linear(in_features = 200, out_features = output_shape),
        )

    def forward(self,x):
        return self.layers(x)


##Weakmodifiedmobilenetv2


class _InvertedResidualBlock(nn.Module): #part of weakmodifiedmobilenetv2
    def __init__(self, input, output, stride, expand_ratio, dropout = 0.1):
        super(_InvertedResidualBlock, self).__init__()

        hidden_units = input * expand_ratio

        self.residual_check = (stride == 1 and input == output)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_units, hidden_units, 3, stride, 1, groups=hidden_units, bias=False),
                nn.Dropout(p = dropout),
                nn.BatchNorm2d(hidden_units),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_units, output, 1, 1, 0, bias=False),
                nn.Dropout(p = dropout),
                nn.BatchNorm2d(output),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(input, hidden_units, 1, 1, 0, bias=False),
                nn.Dropout(p = dropout),
                nn.BatchNorm2d(hidden_units),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_units, hidden_units, 3, stride, 1, groups=hidden_units, bias=False),
                nn.Dropout(p = dropout),
                nn.BatchNorm2d(hidden_units),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_units, output, 1, 1, 0, bias=False),
                nn.Dropout(p = dropout),
                nn.BatchNorm2d(output),
            )

        reduction_ratio = 0.25
        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=output, out_channels=int(output*reduction_ratio), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=int(output*reduction_ratio), out_channels=output, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, input):
        x = self.conv(input)
        if self.residual_check == True:
            return input + x*self.se_layer(x) #adding residual if pass residual check

        else:

            return x*self.se_layer(x)

class WeakModifiedMobileNetv1(nn.Module):
    def __init__(self, color_channels: int, output_features:int, first_input: int, dropout:float = 0.1):
        """A weaker and modified version of Mobilenet, version 1
        Args:
        color_channels: the number of color channels of the input images.
        output_features: the number of output classes.
        first_input: the first input the image get turned into 
        (if first_input = 16 an image of (32,3,32,32) -> (32,16,32,32))
        dropout: the p value of dropout layer
        """
        super().__init__()

        self.dropout = dropout
        self.layers_settings = [ #(output, stride, expand_ratio, num_layers, whether to add pool)
            (16, 1, 1, 1),
            (32, 2, 3, 2),
            (64, 2, 3, 3),
            (96, 2, 3, 3),
            (96, 1, 3, 1),
        ]
        input_channels = first_input
        layers = [
            self.three_conv_bottleneck(input_features = color_channels, output_features = input_channels, stride = 1),
        ]
        blocks =  _InvertedResidualBlock
        for out, stride, ratio, num in self.layers_settings:
            output_channels = out
            for i in range(num):

                block = blocks(input = input_channels, output = output_channels, stride = stride if i == 0 else 1, expand_ratio = ratio)#input, output, stride, expand_ratio
                layers.append(block)
                input_channels = output_channels

        last_layers = nn.Sequential(
            self.first_conv_bottleneck(input_features = output_channels, output_features = output_channels),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features = 96, out_features = output_features)
        )
        layers.append(last_layers)
        self.sequential_layers = nn.Sequential(*layers)
        self._initialize_weights()
    def three_conv_bottleneck(self, input_features, output_features, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels = input_features,
                      out_channels = output_features,
                      kernel_size = 3, #kernal size 3x3
                      stride = stride,
                      padding = 1,
                      bias = False),
            nn.Dropout(p = self.dropout),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
        )

    def first_conv_bottleneck(self, input_features, output_features):
        return nn.Sequential(
            nn.Conv2d(in_channels = input_features,
                      out_channels = output_features,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.Dropout( p =self.dropout),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        return self.sequential_layers(x)
    
# model = WeakModifiedMobileNetv1(color_channels=3, output_features=10, first_input = 32)
# minfo = summary(model = model, input_size = (32,3,32,32))
        
class WeakModifiedMobileNetv2(nn.Module):
    def __init__(self, color_channels, output_features, first_input):
        """A way weaker resnet9 :'( i can't run 7m parameters"""
        super().__init__()

        self.layers_settings = [ #(output, stride, expand_ratio, num_layers, whether to add pool at the end)
            (16, 1, 1, 1, 0),
            (32, 1, 6, 2, 1),
            (64, 1, 6, 3, 1),
            (96, 1, 6, 3, 1),
            (128, 1,2, 1, 0),
        ]
        input_channels = first_input
        layers = [
            self.three_conv_bottleneck(input_features = color_channels, output_features = input_channels, stride = 1),
        ]
        blocks =  _InvertedResidualBlock

        for out, stride, ratio, num, pool in self.layers_settings:
            output_channels = out
            for i in range(num):

                block = blocks(input = input_channels, output = output_channels, stride = stride if i == 0 else 1, expand_ratio = ratio)#input, output, stride, expand_ratio
                layers.append(block)
                input_channels = output_channels
            if pool == 1:
                layers.append(nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7))

        last_layers = nn.Sequential(
            self.first_conv_bottleneck(input_features = output_channels, output_features = output_channels),
            nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7),
            nn.Flatten(),
            nn.Linear(in_features = 6272, out_features = output_features)
        )
        layers.append(last_layers)
        self.sequential_layers = nn.Sequential(*layers)
        self._initialize_weights()
    def three_conv_bottleneck(self, input_features, output_features, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels = input_features,
                      out_channels = output_features,
                      kernel_size = 3, #kernal size 3x3
                      stride = stride,
                      padding = 1,
                      bias = False),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
        )

    def first_conv_bottleneck(self, input_features, output_features):
        return nn.Sequential(
            nn.Conv2d(in_channels = input_features,
                      out_channels = output_features,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
        )
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        return self.sequential_layers(x)

#Resnet9
class ResNet9(nn.Module):
    def __init__(self, color_channels, output_features):
        """A way weaker resnet9 :'( i can't run 7m parameters"""
        super().__init__()
        self.first_layers = nn.Sequential(
            self.conv_block(color_channels, 16,  if_pool = False),
            self.conv_block(16, 32, if_pool = True)
            )

        self.res1 = nn.Sequential(
            self.conv_block(32, 32, if_pool = False),
            self.conv_block(32, 32, if_pool = False)
        )

        self.second_layers = nn.Sequential(
            self.conv_block(32, 64, if_pool = True),
            self.conv_block(64, 64, if_pool = True)
        )

        self.res2 = nn.Sequential(
            self.conv_block(64, 64, if_pool = False),
            self.conv_block(64, 64, if_pool = False)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), #flatten image
            nn.Linear(in_features = 6400, out_features = output_features)
        )
    def conv_block(self, input_features, output_features, kernel_size = 3, stride = 1, if_pool:bool = False):
        layers = [
            nn.Conv2d(in_channels = input_features,
                      out_channels = output_features,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = 1),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
        ]

        if if_pool: #add pool layer
            layers.append(nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layers(x)
        x = self.res1(x) + x
        x = self.second_layers(x)

        x = self.res2(x) +x

        return self.classifier(x)

#Densenet. densenet aint that good. 
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate:float = 0):
        super().__init__()
        self.drop_rate = drop_rate #set the dropout rate
        #init the layers
        # print(bn_size*growth_rate)
        self.first_layers = nn.Sequential(
            nn.BatchNorm2d(int(num_input_features)),
            nn.GELU(),
            nn.Conv2d(in_channels = num_input_features, out_channels = bn_size * growth_rate, kernel_size = 1, stride = 1, bias = False), #1x1 convolution layer
        )

        self.second_layer = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.GELU(),
            nn.Conv2d(in_channels = bn_size * growth_rate, out_channels = growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )

        if drop_rate >0: #another check, dont want some random stuff ruining it.
            self.drop_out = nn.Dropout(p = drop_rate)
    def forward(self, previous_features: List[torch.Tensor]) -> torch.Tensor:
        if isinstance(previous_features, torch.Tensor):
            input = [previous_features]
        else:
            input = previous_features


        output = self.first_layers(torch.cat(input, dim =1)) #catted the inputs, then put them thorugh layers. i can't write the proper pronunciation.

        output = self.second_layer(output)
        if self.drop_rate > 0: #only apply dropout if drop rate > 0
            output = self.drop_out(output)

        return output


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        self.layers = nn.ModuleList([_DenseLayer(num_input_features = num_input_features + i * growth_rate,
                                                growth_rate = growth_rate,
                                                bn_size = bn_size,
                                                drop_rate = drop_rate) for i in range(num_layers)]) #create an iterable of Modules.

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim = 1) #return all output features, catted.

class _TransitionLayers(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
            nn.Conv2d(in_channels = num_input_features, out_channels = num_output_features, kernel_size = 1, stride = 1, bias = False),
            nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7)
        )

    def forward(self, x):
        return self.layer(x)


class WeakModifiedDenseNet(nn.Module):
    def __init__(self, color_channels, blocks_config: List[int], num_init_features, bn_size, growth_rate, drop_rate, num_classes):
        """Args:
        blocks_config: List[int] 'The amount of layers in each dense block, in order.'
        num_init_features: int 'similar to hidden units. Just put sth, the code should handle the rest.'
        bn_size: int 'similar to hidden units. Just put sth, the code should handle the rest.'
        growth_rate: int 'the growth rate after each '
        """
        super().__init__()

        #add the dense block + transition layers
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = color_channels,
                      out_channels = num_init_features,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1, bias = False), #bias isnt needed cause there's batch norm before or after it. batchnorm research paper. probably somewhere in there.
            nn.BatchNorm2d(num_init_features),
            nn.GELU(),
            nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7)
            )

        num_features = num_init_features

        for i, num_layers in enumerate(blocks_config):
            block = _DenseBlock(
                num_layers = num_layers,
                num_input_features = num_features,
                bn_size = bn_size,
                growth_rate = growth_rate,
                drop_rate = drop_rate
            )

            self.layers.add_module(f"dense_block_{i+1}", block)

            num_features += num_layers * growth_rate

            if i != len(blocks_config) - 1:
                transition = _TransitionLayers(num_input_features = num_features, num_output_features = num_features // 2)

                self.layers.add_module(f"transition_layer_{i+1}", transition)
                num_features = num_features // 2

        self.final_layers = nn.Sequential(
            nn.GELU(),
            nn.FractionalMaxPool2d(kernel_size = 2, output_ratio = 0.7),
            nn.Flatten(),
            nn.Linear(in_features = 24800, out_features = 64),
            nn.Linear(in_features = 64, out_features = num_classes)
        )
        for modules in self.modules(): #initialize the modules weights. i copied this from the internet. no idea if these init works good or bad.
            if isinstance(modules, nn.Conv2d):
                nn.init.kaiming_normal_(modules.weight)
            elif isinstance(modules, nn.BatchNorm2d):
                nn.init.constant_(modules.weight, 1) #fill input tensor with a value (1)
                nn.init.constant_(modules.bias, 0) #fill input tensor with a value(0)
    def forward(self, x):
        x = self.layers(x)

        x = self.final_layers(x)
        return x

##some params for densenet. 
# model = WeakModifiedDenseNet(
#     color_channels = 3,
#     blocks_config = [6, 12],
#     num_init_features = 16,
#     bn_size = 4,
#     growth_rate = 16,
#     drop_rate = 0,
#     num_classes = 10
#     ).to(device)


###VIT model. not recommened for self training. just take their params and do some transfer learning. needs too much data and computer resource. 
#creaing a vision transformer

class _PatchEmbd(nn.Module):
    """
    A patch embedding block of a vision trnasformer.
    Args:
    in_channels: the amount of color channels.
    embedding_dim: the embedding dimension.
    patch_size: the size of patch (Height x Width)
    """

    def __init__(self, in_channels: int, embedding_dim, patch_size: Tuple[int, int]) -> None:
        super().__init__()

        self.patch_embd_layers = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = embedding_dim,
                kernel_size = patch_size,
                stride = patch_size,
                padding = 0),
            nn.Flatten(
                start_dim=-2,
                end_dim=-1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embd_layers(x).permute(0,2,1) ##turn image shape into format (batch, num of patches, embeddings dim)

class _PosToken(nn.Module):
    """Create class token, position embedding for ViT."""
    def __init__(self, embedding_dim: int, num_patches) -> None:
        super().__init__()

        self.class_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim),#batch size, number of token, embedding dim
            requires_grad = True
        )

        self.pos_embd = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_dim),#batch size, number of token, embedding dim
            requires_grad = True
        )

    def forward(self, x: torch.Tensor, batch_size) -> torch.Tensor:
        class_token = self.class_token.expand(batch_size, -1, -1)
        return torch.cat((class_token, x), dim = 1) + self.pos_embd

class Vit(nn.Module):
    """
    Create a vision transformer.

    Args:
    in_channels:
    """

    def __init__(self,
                 color_channels: int = 3,
                 image_size: Tuple | int = 32,
                 patch_size: Tuple[int, int] | int = 4,
                 num_transformer_layers: int = 8,
                 mlp_heads = 4,
                 mlp_dropout: float = 0.1,
                 mlp_size: int = 1024,
                 num_classes: int = 10
                 ):

        super().__init__()

        if type(image_size) == int: #process image_size input
            image_height = image_size
            image_width = image_size
        else:
            image_height, image_width = image_size

        if type(patch_size) == int: #process patch size input
            patch_height = patch_size
            patch_width = patch_size
        else:
            patch_height, patch_width = patch_size

        assert image_height % patch_height == 0, "Image height needs to be divisible by patch height"
        assert image_width % patch_width == 0, "Image width needs to be divisible by patch width"

        self.embeddings = (patch_height * patch_width) * color_channels
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.embeddings, #hidden size D, or embeddings dim, or patch_h x patch_w * color channels
            nhead = mlp_heads, #number of heads in multihead self attention
            dim_feedforward = mlp_size, #mlp size, or the hidden units in mlp block
            dropout = mlp_dropout, #the dropout for mlp layers
            activation = 'gelu', #type of non_linear activation,
            batch_first = True, #do batches come first?
            norm_first = True,  #if True, layer norm is done prior to attention and feedforward operations, respectively. Otherwise it’s done after. Default: False
        )

        self.pos_token = _PosToken(embedding_dim = self.embeddings, num_patches = self.num_patches)

        self.patch_embd = _PatchEmbd(in_channels = color_channels, embedding_dim = self.embeddings, patch_size = [patch_height, patch_width])

        self.transformer_encoder_blocks = nn.TransformerEncoder(
            encoder_layer = transformer_encoder_layer,
            num_layers = num_transformer_layers
        ) #stack many transformer encoder layer together.

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.embeddings),
            nn.Linear(in_features = self.embeddings, out_features = num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.patch_embd(x)

        x = self.pos_token(x, batch_size) #postion embedding and class token. 

        x = self.transformer_encoder_blocks(x)

        x = self.classifier(x[:,0]) #put 0 logit index through classifier

        return x

# vit_model = Vit(
#     color_channels = 3,
#     image_size = 32,
#     patch_size = 8,
#     num_transformer_layers = 12,
#     mlp_heads = 8,
#     mlp_size = 3072
# ).to(device)
# summary = utils.get_model_summary(model = vit_model, input_size = (BATCH_SIZE,3,32,32))
# print(summary)

# dummy_img  = torch.randn(BATCH_SIZE, 3, 32, 32).to(device)
# dummy_preds = vit_model(dummy_img)
# print(f"Dummy predictions shape: {dummy_preds.shape}")
# print(f"First dummy prediction: {dummy_preds[0]}")

#create optimizer and loss function
# optimizer = torch.optim.Adam(
#     params = vit_model.parameters(), 
#     lr = 1e-3,
#     weight_decay = 0.3
#     )
# loss_fn = nn.CrossEntropyLoss()
# writer = utils.create_writer(experiment_name = 'Base Vision Transformer', model_name = 'weak, modified ViT base') #create tensorboard writer
