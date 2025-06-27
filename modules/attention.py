import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class OmniSRAttention(nn.Module):
    """
    Omni-SR lightweight attention mechanism.
    Combines spatial and channel attention efficiently.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(channels // reduction, 1)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial))
        
        # Combine attention
        x = x * channel_out * spatial_out
        return x

class SpatialChannelAttention(nn.Module):
    """
    Enhanced spatial and channel attention mechanism.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction, 1)
        
        # Channel attention with SE-style
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention with CBAM-style
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x = x * spatial_weights
        
        return x

class RegionAdaptiveAttention(nn.Module):
    """
    Region-adaptive attention for selective feature enhancement.
    Inspired by Swift attention for efficient processing.
    """
    def __init__(self, channels: int, num_regions: int = 4, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.num_regions = num_regions
        reduced_channels = max(channels // reduction, 1)
        
        # Region-wise attention
        self.region_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, reduced_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channels, channels, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_regions)
        ])
        
        # Global attention - use spatial attention instead of global pooling
        self.global_attention = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Region fusion
        self.region_fusion = nn.Conv2d(channels * (num_regions * 2), channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        original_h = h
        
        # Ensure height is divisible by num_regions
        if h % self.num_regions != 0:
            # Pad to make it divisible
            pad_h = self.num_regions - (h % self.num_regions)
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')
            h = x.shape[2]
        
        # Split into regions
        region_size = h // self.num_regions
        regions = []
        
        for i in range(self.num_regions):
            start_h = i * region_size
            end_h = (i + 1) * region_size
            region = x[:, :, start_h:end_h, :]
            region_att = self.region_attention[i](region)
            regions.append(region * region_att)
        
        # Global attention
        global_att = self.global_attention(x)
        global_out = x * global_att
        for i in range(self.num_regions):
            start_h = i * region_size
            end_h = (i + 1) * region_size
            regions.append(global_out[:, :, start_h:end_h, :])
        
        # Concatenate and fuse
        concat_features = torch.cat(regions, dim=1)
        output = self.region_fusion(concat_features)

        # 恢复到原始高度
        output = F.interpolate(output, size=(original_h, w), mode='bilinear', align_corners=False)
        return output

class EfficientAttentionBlock(nn.Module):
    """
    Efficient attention block combining multiple attention mechanisms.
    """
    def __init__(self, channels: int, attention_type: str = 'omni'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'omni':
            self.attention = OmniSRAttention(channels)
        elif attention_type == 'spatial_channel':
            self.attention = SpatialChannelAttention(channels)
        elif attention_type == 'region_adaptive':
            self.attention = RegionAdaptiveAttention(channels)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
    def forward(self, x):
        return self.attention(x) 