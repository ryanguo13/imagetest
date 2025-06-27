# backbone.py
"""
Backbone network components: PixelShuffle, BSConv, ConvNeXt-style blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PixelShuffleBlock(nn.Module):
    """
    PixelShuffle upsampling block for super-resolution.
    Replaces bicubic interpolation with learnable upsampling.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class BSConvBlock(nn.Module):
    """
    BSConv (Block Separable Convolution) for lightweight design.
    Combines depthwise and pointwise convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style residual block with modern design principles.
    Uses depthwise conv, layer norm, and GELU activation.
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(torch.ones(1))
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(self.gamma * x)
        return x

class LightweightBackbone(nn.Module):
    """
    Lightweight backbone combining BSConv and ConvNeXt blocks.
    Designed for super-resolution without bicubic interpolation.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_blocks: int = 8, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_in = BSConvBlock(in_channels, base_channels)
        
        # Main body with ConvNeXt blocks
        self.body = nn.ModuleList([
            ConvNeXtBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # Upsampling with PixelShuffle
        self.upscale = nn.ModuleList()
        current_channels = base_channels
        
        # Calculate how many upsampling steps we need
        # For scale_factor = 4, we need 2 steps of 2x upsampling
        # For scale_factor = 2, we need 1 step of 2x upsampling
        # For scale_factor = 8, we need 3 steps of 2x upsampling
        upscale_steps = 0
        temp_scale = scale_factor
        while temp_scale > 1:
            upscale_steps += 1
            temp_scale //= 2
        
        for i in range(upscale_steps):
            if i == upscale_steps - 1:
                # Last upsampling step
                self.upscale.append(
                    PixelShuffleBlock(current_channels, in_channels, 2)
                )
            else:
                # Intermediate upsampling steps
                self.upscale.append(
                    PixelShuffleBlock(current_channels, current_channels // 2, 2)
                )
                current_channels = current_channels // 2
        
        # If no upsampling needed (scale_factor = 1), add a final conv
        if upscale_steps == 0:
            self.conv_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = None
        
    def forward(self, x):
        # Feature extraction
        feat = self.conv_in(x)
        
        # Main body
        for block in self.body:
            feat = block(feat)
        
        # Upsampling
        for up in self.upscale:
            feat = up(feat)
        
        # Final output (if needed)
        if self.conv_out is not None:
            feat = self.conv_out(feat)
        
        return feat 