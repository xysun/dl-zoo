import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import math
from collections import OrderedDict

'''
from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
adapted for 32x32 (`small_inputs=True`) and always set `efficient=True`
also use relu
'''

def _bn_function_factory(norm, relu, conv):
    # we need this because torch.checkpoint requires a forward function
    def bottleneck_function(*inputs):
        concatenated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output
    
    return bottleneck_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()

        # bottleneck layer: expand to bn_size*x
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate * bn_size, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(growth_rate * bn_size))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(growth_rate * bn_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, *prev_features): # concatenate magic
        bottleneck_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bottleneck_function, *prev_features)
        else:
            bottleneck_output = bottleneck_function(*prev_features)
        
        # if self.drop_rate > 0:
        #     new_features = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
        # else:
        #     new_features = bottleneck_output
        
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, 
                growth_rate, 
                bn_size, 
                drop_rate
            )
            self.add_module('denselayer%d' % (i+1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features) # concat happens here!
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    # transition: norm => relu => conv 1x1 => pool
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    '''
    1 DenseNet = 3 DenseBlock;
    1 DenseBlock = `l` DenseLayer;
    1 DenseLayer = 1x1 conv => 3x3 conv => dropout
    each conv = batchnorm => relu => actual conv
    '''


    def __init__(self, growth_rate=12, block_config=(16,16,16), 
        bn_size=4,
        drop_rate=0.2, num_classes=100
        ):

        num_init_features = 2 * growth_rate

        assert len(block_config) == 3, 'must have 3 dense blocks'

        super(DenseNet, self).__init__()
        
        # first convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        ]))

        # each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers, 
                num_features,
                bn_size, 
                growth_rate, 
                drop_rate
            )
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                # transition 
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2
            
        # final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # linear classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1)).view(features.size(0), -1) # global avg pool
        out = self.classifier(out)
        return out
