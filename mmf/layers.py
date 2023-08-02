from typing import Optional

import torch
from mmf.registry import registry
from mmf.decoders import LanguageDecoder
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class ConvNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding_size="same",
            pool_stride=2,
            batch_norm=True,
    ):
        super().__init__()

        if padding_size=="same":
            padding_size = kernel_size // 2
        
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
        self.max_pool2d = nn.max_pool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norm_2d = nn.batch_norm(out_channels)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))

        if self.batch_norm:
            x = self.batch_norm_2d(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, input):
        if input.dim() > 1:
            input = input.view(input.size(0), -1)

        return input
    
class UnFlatten(nn.Module):
    def forward(self, input, sizes=None):
        if sizes is None:
            sizes=[]
        return input.view(input.size(0), *sizes)
    
class GatedTanh(nn.Module):
    """
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\\in R^m => y \\in R^n
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \\circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))

        #Element wise multiplication
        y = y_tilda * gated

        return y
    
class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)