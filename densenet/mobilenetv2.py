'''
MobileNetV2 implementation
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

class BottleNeck(nn.Module):
    '''
    inverted residual block; thin => thick => thin
    where the middle convolution is done using a depthwise convolution to save parameters
    t: expansion factor on number of input filters
    width_multiplier (alpha): 

    input | operator | output
    h * w * k | 1x1 conv2d, relu6 | h * w * (t*alpha*k)

    batch normalization after every layer

    '''
    def __init__(self, input_channel, output_channel, stride, t=1):
        # input_channel and output_channel here is after width_multiplier
        super(BottleNeck, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride

        conv1_output_channel = int(t*input_channel)
        self.conv1 = nn.Conv2d(input_channel, conv1_output_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(conv1_output_channel)

        # setting groups = input_channel for the depth-wise convolution
        self.conv2 = nn.Conv2d(conv1_output_channel, conv1_output_channel, stride=stride, kernel_size=3, groups=conv1_output_channel, padding=1)
        self.bn2 = nn.BatchNorm2d(conv1_output_channel)

        # final 1x1 conv2d
        self.conv3 = nn.Conv2d(conv1_output_channel, output_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace=True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))

        # add shortcut if possible
        if self.input_channel == self.output_channel and self.stride == 1:
            # no downsampling when stride is 1 
            try:
                x = inputs + x
            except RuntimeError:
                print(x.shape, inputs.shape, self.input_channel)
                sys.exit(1)
        
        return x
        

class MobileNetV2(nn.Module):
    def __init__(self, n_classes=10, width_multiplier=1.):
        super(MobileNetV2, self).__init__()
        
        # first conv
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=int(32*width_multiplier), kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(int(32*width_multiplier))
        # bottleneck layers
        config = [
            (1,16,1,1),
            (6,24,2,1),
            (6,32,3,1),
            (6,64,4,2),
            (6,96,3,1),
            (6,160,3,2),
            (6,320,1,1)
        ]
        bottlenecks = []
        input_channel = int(32 * width_multiplier)
        for t, c, n, s in config:
            for i in range(n):
                output_channel = int(c*width_multiplier)
                if i > 0:
                    stride = 1
                else:
                    stride = s
                layer = BottleNeck(input_channel, output_channel, stride, t)
                input_channel = output_channel
                bottlenecks.append(layer)
        self.bottlenecks = nn.Sequential(*bottlenecks)
        # last layer
        self.conv1 = nn.Conv2d(int(320*width_multiplier), 1280, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, n_classes)
        

    def forward(self, inputs):
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = nn.Dropout(0.2)(x)
        x = self.fc(x)

        return x
