#!/usr/bin/env python3
"""
Basic functionality tests for the lightweight super-resolution project.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.backbone import LightweightBackbone, PixelShuffleBlock, BSConvBlock, ConvNeXtBlock
from modules.attention import OmniSRAttention, SpatialChannelAttention, RegionAdaptiveAttention
from modules.lora import ConvLoRA, LoRALinear
from modules.transformer import ProgressiveFocusTransformer, LightweightTransformer
from modules.gan import GANSplice, AdversarialLoss

def test_backbone():
    """Test backbone modules."""
    print("Testing backbone modules...")
    
    # Test PixelShuffleBlock
    pixel_shuffle = PixelShuffleBlock(64, 32, scale_factor=2)
    x = torch.randn(1, 64, 32, 32)
    output = pixel_shuffle(x)
    assert output.shape == (1, 32, 64, 64), f"PixelShuffle output shape: {output.shape}"
    print("✓ PixelShuffleBlock passed")
    
    # Test BSConvBlock
    bsconv = BSConvBlock(64, 128)
    x = torch.randn(1, 64, 32, 32)
    output = bsconv(x)
    assert output.shape == (1, 128, 32, 32), f"BSConv output shape: {output.shape}"
    print("✓ BSConvBlock passed")
    
    # Test ConvNeXtBlock
    convnext = ConvNeXtBlock(64)
    x = torch.randn(1, 64, 32, 32)
    output = convnext(x)
    assert output.shape == (1, 64, 32, 32), f"ConvNeXt output shape: {output.shape}"
    print("✓ ConvNeXtBlock passed")
    
    # Test LightweightBackbone
    backbone = LightweightBackbone(in_channels=3, base_channels=32, num_blocks=4, scale_factor=4)
    x = torch.randn(1, 3, 64, 64)
    output = backbone(x)
    assert output.shape == (1, 3, 256, 256), f"Backbone output shape: {output.shape}"
    print("✓ LightweightBackbone passed")

def test_attention():
    """Test attention modules."""
    print("\nTesting attention modules...")
    
    # Test OmniSRAttention
    omni_attention = OmniSRAttention(64)
    x = torch.randn(1, 64, 32, 32)
    output = omni_attention(x)
    assert output.shape == (1, 64, 32, 32), f"OmniSR output shape: {output.shape}"
    print("✓ OmniSRAttention passed")
    
    # Test SpatialChannelAttention
    spatial_attention = SpatialChannelAttention(64)
    x = torch.randn(1, 64, 32, 32)
    output = spatial_attention(x)
    assert output.shape == (1, 64, 32, 32), f"SpatialChannel output shape: {output.shape}"
    print("✓ SpatialChannelAttention passed")
    
    # Test RegionAdaptiveAttention
    region_attention = RegionAdaptiveAttention(64, num_regions=4)
    x = torch.randn(1, 64, 32, 32)
    output = region_attention(x)
    assert output.shape == (1, 64, 32, 32), f"RegionAdaptive output shape: {output.shape}"
    print("✓ RegionAdaptiveAttention passed")

def test_lora():
    """Test LoRA modules."""
    print("\nTesting LoRA modules...")
    
    # Test ConvLoRA
    conv_lora = ConvLoRA(64, 128, rank=4)
    x = torch.randn(1, 64, 32, 32)
    output = conv_lora(x)
    assert output.shape == (1, 128, 32, 32), f"ConvLoRA output shape: {output.shape}"
    print("✓ ConvLoRA passed")
    
    # Test LoRALinear
    linear_lora = LoRALinear(64, 128, rank=4)
    x = torch.randn(1, 64)
    output = linear_lora(x)
    assert output.shape == (1, 128), f"LoRALinear output shape: {output.shape}"
    print("✓ LoRALinear passed")

def test_transformer():
    """Test transformer modules."""
    print("\nTesting transformer modules...")
    
    # Test ProgressiveFocusTransformer
    pft = ProgressiveFocusTransformer(64, num_heads=8, num_focus_stages=2)
    x = torch.randn(1, 64, 32, 32)
    # Reshape for transformer
    x = x.flatten(2).transpose(1, 2)  # (1, 1024, 64)
    output = pft(x)
    assert output.shape == (1, 1024, 64), f"PFT output shape: {output.shape}"
    print("✓ ProgressiveFocusTransformer passed")
    
    # Test LightweightTransformer
    lw_transformer = LightweightTransformer(64)
    x = torch.randn(1, 64, 32, 32)
    output = lw_transformer(x)
    assert output.shape == (1, 64, 32, 32), f"LightweightTransformer output shape: {output.shape}"
    print("✓ LightweightTransformer passed")

def test_gan():
    """Test GAN modules."""
    print("\nTesting GAN modules...")
    
    # Test GANSplice
    gan_splice = GANSplice(3, discriminator_type='patch')
    fake_images = torch.randn(1, 3, 128, 128)
    real_images = torch.randn(1, 3, 128, 128)
    
    fake_logits, real_logits = gan_splice(fake_images, real_images)
    assert fake_logits.shape[0] == 1, f"Fake logits shape: {fake_logits.shape}"
    assert real_logits.shape[0] == 1, f"Real logits shape: {real_logits.shape}"
    print("✓ GANSplice passed")
    
    # Test AdversarialLoss
    adv_loss = AdversarialLoss('lsgan')
    loss = adv_loss(fake_logits, real_logits, is_discriminator=False)
    assert isinstance(loss, torch.Tensor), f"Adversarial loss type: {type(loss)}"
    print("✓ AdversarialLoss passed")

def test_model_integration():
    """Test model integration."""
    print("\nTesting model integration...")
    
    # Create a simple integrated model
    class SimpleIntegratedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = LightweightBackbone(in_channels=3, base_channels=32, num_blocks=2, scale_factor=2)
            self.attention = OmniSRAttention(3)
            
        def forward(self, x):
            x = self.backbone(x)
            x = self.attention(x)
            return x
    
    model = SimpleIntegratedModel()
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    assert output.shape == (1, 3, 128, 128), f"Integrated model output shape: {output.shape}"
    print("✓ Model integration passed")

def test_device_compatibility():
    """Test device compatibility."""
    print("\nTesting device compatibility...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test on device
    backbone = LightweightBackbone(in_channels=3, base_channels=32, num_blocks=2, scale_factor=2)
    backbone.to(device)
    
    x = torch.randn(1, 3, 64, 64).to(device)
    output = backbone(x)
    assert output.shape == (1, 3, 128, 128), f"Device test output shape: {output.shape}"
    print("✓ Device compatibility passed")

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Lightweight Super-Resolution Tests")
    print("=" * 50)
    
    try:
        test_backbone()
        test_attention()
        test_lora()
        test_transformer()
        test_gan()
        test_model_integration()
        test_device_compatibility()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 