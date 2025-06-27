import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConvLoRA(nn.Module):
    """
    Convolutional Low-Rank Adaptation (ConvLoRA) for efficient fine-tuning.
    Reduces parameters while maintaining performance.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 rank: int = 4, alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size, 
                               padding=kernel_size//2, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, 1, bias=False)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Low-rank adaptation
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = self.dropout(lora_out)
        return lora_out * self.scaling

class LoRALinear(nn.Module):
    """
    Linear Low-Rank Adaptation for fully connected layers.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, 
                 alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Low-rank adaptation
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = self.dropout(lora_out)
        return lora_out * self.scaling

class LoRAAdapter(nn.Module):
    """
    LoRA adapter for existing models.
    Can be added to any layer for efficient fine-tuning.
    """
    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Determine layer type and create appropriate LoRA
        if isinstance(base_layer, nn.Conv2d):
            self.lora = ConvLoRA(
                base_layer.in_channels, base_layer.out_channels,
                base_layer.kernel_size[0], rank, alpha
            )
        elif isinstance(base_layer, nn.Linear):
            self.lora = LoRALinear(
                base_layer.in_features, base_layer.out_features,
                rank, alpha
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(base_layer)}")
            
    def forward(self, x):
        # Base layer output
        base_out = self.base_layer(x)
        
        # LoRA adaptation
        lora_out = self.lora(x)
        
        return base_out + lora_out

class DistillationLoss(nn.Module):
    """
    Distillation loss for knowledge transfer from teacher to student.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_output, teacher_output, target):
        # Knowledge distillation loss
        student_logits = F.log_softmax(student_output / self.temperature, dim=1)
        teacher_logits = F.softmax(teacher_output / self.temperature, dim=1)
        distill_loss = self.kl_loss(student_logits, teacher_logits) * (self.temperature ** 2)
        
        # Task loss (e.g., MSE for super-resolution)
        task_loss = F.mse_loss(student_output, target)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return total_loss, distill_loss, task_loss

class FeatureDistillation(nn.Module):
    """
    Feature-level distillation for intermediate representations.
    """
    def __init__(self, feature_layers: list, weights: Optional[list] = None):
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights if weights is not None else [1.0] * len(feature_layers)
        
    def forward(self, student_features, teacher_features):
        """
        student_features: list of student feature maps
        teacher_features: list of teacher feature maps
        """
        distill_loss = 0.0
        
        for i, (s_feat, t_feat, weight) in enumerate(zip(student_features, teacher_features, self.weights)):
            # L2 loss between feature maps
            layer_loss = F.mse_loss(s_feat, t_feat)
            distill_loss += weight * layer_loss
            
        return distill_loss

# Import math for initialization
import math 