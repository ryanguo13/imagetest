# gan.py
"""
Lightweight adversarial loss (GAN-splice) for texture realism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Discriminator(nn.Module):
    """
    Lightweight discriminator for GAN training.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 num_layers: int = 4):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(current_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_channels = out_channels
        
        # Final layer
        layers.append(nn.Conv2d(current_channels, 1, 4, 1, 1))
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.discriminator(x)

class PatchDiscriminator(nn.Module):
    """
    Patch-based discriminator for local texture discrimination.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        )
        
    def forward(self, x):
        return self.features(x)

class GANSplice(nn.Module):
    """
    GAN-splice: Lightweight adversarial loss for texture realism.
    """
    def __init__(self, in_channels: int = 3, discriminator_type: str = 'patch'):
        super().__init__()
        self.discriminator_type = discriminator_type
        
        if discriminator_type == 'patch':
            self.discriminator = PatchDiscriminator(in_channels)
        else:
            self.discriminator = Discriminator(in_channels)
            
    def forward(self, fake_images, real_images=None):
        """
        Forward pass for GAN training.
        Args:
            fake_images: Generated images from the generator
            real_images: Ground truth images (optional, for training)
        """
        fake_logits = self.discriminator(fake_images)
        
        if real_images is not None:
            real_logits = self.discriminator(real_images)
            return fake_logits, real_logits
        else:
            return fake_logits

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    """
    def __init__(self, loss_type: str = 'vanilla', target_real_label: float = 1.0, 
                 target_fake_label: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None  # WGAN uses different loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def forward(self, fake_logits, real_logits=None, is_discriminator: bool = False):
        """
        Compute adversarial loss.
        Args:
            fake_logits: Discriminator output for fake images
            real_logits: Discriminator output for real images
            is_discriminator: Whether computing loss for discriminator
        """
        if self.loss_type == 'vanilla':
            if is_discriminator:
                # Discriminator loss
                real_loss = self.criterion(real_logits, 
                                         torch.ones_like(real_logits) * self.target_real_label)
                fake_loss = self.criterion(fake_logits, 
                                         torch.zeros_like(fake_logits) * self.target_fake_label)
                return (real_loss + fake_loss) * 0.5
            else:
                # Generator loss
                return self.criterion(fake_logits, 
                                    torch.ones_like(fake_logits) * self.target_real_label)
        elif self.loss_type == 'lsgan':
            if is_discriminator:
                real_loss = self.criterion(real_logits, 
                                         torch.ones_like(real_logits) * self.target_real_label)
                fake_loss = self.criterion(fake_logits, 
                                         torch.zeros_like(fake_logits) * self.target_fake_label)
                return (real_loss + fake_loss) * 0.5
            else:
                return self.criterion(fake_logits, 
                                    torch.ones_like(fake_logits) * self.target_real_label)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    """
    def __init__(self, feature_layers: list = [2, 7, 12, 21, 30], 
                 weights: Optional[list] = None):
        super().__init__()
        
        try:
            # Try to load VGG19 features
            vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
            self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:31])
            
            # Freeze VGG parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
            self.available = True
            print("PerceptualLoss: VGG19 model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load VGG19 model for perceptual loss: {e}")
            print("PerceptualLoss will be disabled. Training will continue with MSE loss only.")
            self.feature_extractor = None
            self.available = False
            
        self.feature_layers = feature_layers
        self.weights = weights if weights is not None else [1.0] * len(feature_layers)
        
    def forward(self, fake_images, real_images):
        """
        Compute perceptual loss between fake and real images.
        """
        if not self.available:
            # Return zero loss if VGG is not available
            return torch.tensor(0.0, device=fake_images.device, requires_grad=True)
        
        # Normalize images to VGG input range
        fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]
        real_images = (real_images + 1) / 2
        
        # Extract features
        fake_features = self.extract_features(fake_images)
        real_features = self.extract_features(real_images)
        
        # Compute loss
        loss = 0.0
        for fake_feat, real_feat, weight in zip(fake_features, real_features, self.weights):
            loss += weight * F.mse_loss(fake_feat, real_feat)
            
        return loss
        
    def extract_features(self, x):
        """
        Extract features from specified layers.
        """
        if not self.available:
            return []
            
        features = []
        current_layer = 0
        
        for layer in self.feature_extractor:
            x = layer(x)
            current_layer += 1
            
            if current_layer - 1 in self.feature_layers:
                features.append(x)
                
        return features

class StyleLoss(nn.Module):
    """
    Style loss for texture preservation.
    """
    def __init__(self, feature_layers: list = [2, 7, 12, 21, 30]):
        super().__init__()
        self.feature_layers = feature_layers
        self.perceptual_loss = PerceptualLoss(feature_layers)
        
    def gram_matrix(self, x):
        """
        Compute Gram matrix for style representation.
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
        
    def forward(self, fake_images, real_images):
        """
        Compute style loss.
        """
        if not self.perceptual_loss.available:
            # Return zero loss if VGG is not available
            return torch.tensor(0.0, device=fake_images.device, requires_grad=True)
            
        fake_features = self.perceptual_loss.extract_features((fake_images + 1) / 2)
        real_features = self.perceptual_loss.extract_features((real_images + 1) / 2)
        
        style_loss = 0.0
        for fake_feat, real_feat in zip(fake_features, real_features):
            fake_gram = self.gram_matrix(fake_feat)
            real_gram = self.gram_matrix(real_feat)
            style_loss += F.mse_loss(fake_gram, real_gram)
            
        return style_loss 