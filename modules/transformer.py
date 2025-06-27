# transformer.py
"""
Progressive Focus Transformer (PFT) block for non-local learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ProgressiveFocusTransformer(nn.Module):
    """
    Progressive Focus Transformer (PFT) for selective attention.
    Gradually focuses on more important regions during processing.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, num_focus_stages: int = 3):
        super().__init__()
        self.dim = dim
        self.num_focus_stages = num_focus_stages
        
        # Multi-stage attention with progressive focus
        self.attention_stages = nn.ModuleList([
            MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
            for _ in range(num_focus_stages)
        ])
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Focus weights for progressive attention
        self.focus_weights = nn.Parameter(torch.ones(num_focus_stages))
        
        # Drop path
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)
        
    def forward(self, x):
        # Progressive focus attention
        attn_out = 0
        for i, attention in enumerate(self.attention_stages):
            stage_out = attention(self.norm1(x))
            attn_out += self.focus_weights[i] * stage_out
        
        # Residual connection
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class SpatialTransformerBlock(nn.Module):
    """
    Spatial transformer block for 2D feature maps.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        
        # Spatial attention
        self.attention = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Drop path
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)
        
    def forward(self, x):
        # x: (B, H*W, C)
        # Self-attention
        x = x + self.drop_path(self.attention(self.norm1(x)))
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class TransformerModule(nn.Module):
    """
    Complete transformer module for super-resolution.
    Combines spatial and progressive focus attention.
    """
    def __init__(self, channels: int, height: int, width: int, 
                 num_heads: int = 8, num_blocks: int = 2, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ProgressiveFocusTransformer(channels, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x

class LightweightTransformer(nn.Module):
    """
    Lightweight transformer for efficient non-local learning.
    """
    def __init__(self, channels: int, reduction: int = 16, num_heads: int = 2):
        super().__init__()
        self.channels = channels
        self.reduced_channels = max(channels // reduction, 8)  # 确保最小通道数
        
        # Channel reduction
        self.channel_reduce = nn.Conv2d(channels, self.reduced_channels, 1)
        
        # Lightweight attention
        self.attention = MultiHeadAttention(self.reduced_channels, num_heads)
        
        # Channel expansion
        self.channel_expand = nn.Conv2d(self.reduced_channels, channels, 1)
        
        # Layer norm
        self.norm = nn.LayerNorm(self.reduced_channels)
        
    def forward(self, x):
        # Channel reduction
        reduced = self.channel_reduce(x)
        
        # Reshape for attention
        B, C, H, W = reduced.shape
        reduced = reduced.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply attention
        attended = self.attention(self.norm(reduced))
        
        # Reshape back
        attended = attended.transpose(1, 2).reshape(B, C, H, W)
        
        # Channel expansion
        output = self.channel_expand(attended)
        
        return x + output 