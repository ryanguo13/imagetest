# train.py
"""
Training logic, optimizer setup (AdamW, Adan), distillation supervision.
Enhanced with GPU optimization features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os
import time
from tqdm import tqdm
import logging

# Import mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Import our modules
from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock
from modules.lora import ConvLoRA, DistillationLoss, FeatureDistillation
from modules.transformer import LightweightTransformer
from modules.gan import GANSplice, AdversarialLoss, PerceptualLoss

class AdanOptimizer(optim.Optimizer):
    """
    Adan optimizer implementation for faster convergence.
    """
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, 
                 weight_decay=0.01, max_grad_norm=None, no_prox=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       max_grad_norm=max_grad_norm, no_prox=no_prox)
        super().__init__(params, defaults)
        
    def __getstate__(self):
        return super().__getstate__()
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adan does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_diff'] = torch.zeros_like(p.data)
                    state['pre_grad'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq, exp_avg_diff, pre_grad = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff'], state['pre_grad']
                beta1, beta2, beta3 = group['betas']
                
                state['step'] += 1
                
                # Compute diff
                diff = grad - pre_grad
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)
                exp_avg_sq.mul_(beta3).add_((grad + beta2 * diff) ** 2, alpha=1 - beta3)
                
                # Update pre_grad
                pre_grad.copy_(grad)
                
                # Compute step size
                step_size = group['lr']
                
                # Compute bias correction terms
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                bias_correction3 = 1.0 - beta3 ** state['step']
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction3 ** 0.5)).add_(group['eps'])
                
                # Compute update
                update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2) / denom
                
                # Apply weight decay
                if group['weight_decay'] > 0.0:
                    if group['no_prox']:
                        p.data.add_(update, alpha=-step_size)
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                    else:
                        p.data.add_(update, alpha=-step_size)
                        p.data.div_(1 + group['lr'] * group['weight_decay'])
                else:
                    p.data.add_(update, alpha=-step_size)
                    
        return loss

class MultiStageTrainer:
    """
    Multi-stage training with warm-start strategy and GPU optimizations.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU优化设置
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        self.compile_model = config.get('compile_model', False)
        
        # 初始化混合精度训练
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state (must be initialized before model building)
        self.current_stage = 0
        self.best_psnr = 0.0
        
        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 模型编译优化 (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            print("Compiling model for optimization...")
            self.model = torch.compile(self.model, mode='max-autotune')
        
        # Initialize optimizers
        self.optimizer = self._build_optimizer()
        
        # Initialize losses
        self.criterion = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss() if config.get('use_perceptual', False) else None
        
        # GPU监控
        self.gpu_stats = {'peak_memory': 0, 'current_memory': 0}
        
    def _build_model(self):
        """Build model based on current stage."""
        base_channels = self.config['base_channels']
        num_blocks = self.config['num_blocks']
        
        # Stage 0: Basic backbone
        if self.current_stage == 0:
            return LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=self.config['scale_factor']
            )
        
        # Stage 1: Add attention
        elif self.current_stage == 1:
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=self.config['scale_factor']
            )
            # Add attention blocks
            for i, block in enumerate(model.body):
                if i % 2 == 0:  # Add attention every other block
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    # Insert attention after ConvNeXt block
                    model.body[i] = nn.Sequential(block, attention)
            return model
        
        # Stage 2: Add transformer
        elif self.current_stage == 2:
            # Build stage 1 model first
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=self.config['scale_factor']
            )
            # Add attention blocks (stage 1 features)
            for i, block in enumerate(model.body):
                if i % 2 == 0:  # Add attention every other block
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    # Insert attention after ConvNeXt block
                    model.body[i] = nn.Sequential(block, attention)
            
            # Add transformer module (stage 2 feature)
            transformer = LightweightTransformer(base_channels)
            # Insert transformer in the middle
            mid_idx = len(model.body) // 2
            model.body.insert(mid_idx, transformer)
            return model
        
        # Stage 3: Add GAN components
        elif self.current_stage == 3:
            # Build stage 2 model first
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=self.config['scale_factor']
            )
            # Add attention blocks (stage 1 features)
            for i, block in enumerate(model.body):
                if i % 2 == 0:  # Add attention every other block
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    # Insert attention after ConvNeXt block
                    model.body[i] = nn.Sequential(block, attention)
            
            # Add transformer module (stage 2 feature)
            transformer = LightweightTransformer(base_channels)
            # Insert transformer in the middle
            mid_idx = len(model.body) // 2
            model.body.insert(mid_idx, transformer)
            
            # Add GAN discriminator (stage 3 feature)
            self.discriminator = GANSplice(3, 'patch').to(self.device)
            return model
            
        else:
            # Fallback for unknown stages - return stage 3 model
            raise ValueError(f"Unknown training stage: {self.current_stage}. Supported stages are 0-3.")
            
    def _build_optimizer(self):
        """Build optimizer based on config."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adan':
            return AdanOptimizer(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _update_gpu_stats(self):
        """更新GPU内存统计"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3  # GB
            peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.gpu_stats['current_memory'] = current
            self.gpu_stats['peak_memory'] = max(self.gpu_stats['peak_memory'], peak)
            
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch with GPU optimizations."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
            lr_imgs, hr_imgs = lr_imgs.to(self.device, non_blocking=True), hr_imgs.to(self.device, non_blocking=True)
            
            # 混合精度训练
            if self.use_mixed_precision:
                with autocast():
                    # Forward pass
                    sr_imgs = self.model(lr_imgs)
                    
                    # Compute losses
                    mse_loss = self.criterion(sr_imgs, hr_imgs)
                    total_loss = mse_loss
                    
                    # Add perceptual loss if enabled
                    if self.perceptual_loss is not None:
                        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
                        total_loss += self.config.get('perceptual_weight', 0.1) * perceptual_loss
                    
                    # Add adversarial loss if in GAN stage
                    if self.current_stage == 3 and hasattr(self, 'discriminator'):
                        fake_logits = self.discriminator(sr_imgs)
                        adversarial_loss = AdversarialLoss('lsgan')(fake_logits, is_discriminator=False)
                        total_loss += self.config.get('adversarial_weight', 0.001) * adversarial_loss
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准精度训练
                sr_imgs = self.model(lr_imgs)
                
                # Compute losses
                mse_loss = self.criterion(sr_imgs, hr_imgs)
                total_loss = mse_loss
                
                # Add perceptual loss if enabled
                if self.perceptual_loss is not None:
                    perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
                    total_loss += self.config.get('perceptual_weight', 0.1) * perceptual_loss
                
                # Add adversarial loss if in GAN stage
                if self.current_stage == 3 and hasattr(self, 'discriminator'):
                    fake_logits = self.discriminator(sr_imgs)
                    adversarial_loss = AdversarialLoss('lsgan')(fake_logits, is_discriminator=False)
                    total_loss += self.config.get('adversarial_weight', 0.001) * adversarial_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
            
            # 更新GPU统计
            self._update_gpu_stats()
            
            # 计算GPU利用率 (简化版本)
            if batch_idx % 50 == 0:
                gpu_util = min(100, (self.gpu_stats['current_memory'] / 8.0) * 100)  # 假设8GB显存
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'GPU_Mem': f'{self.gpu_stats["current_memory"]:.1f}GB',
                    'GPU_Util': f'{gpu_util:.0f}%'
                })
            
        epoch_time = time.time() - start_time
        samples_per_sec = len(train_loader.dataset) / epoch_time
        
        print(f"Epoch {epoch} - {samples_per_sec:.1f} samples/sec, Peak GPU: {self.gpu_stats['peak_memory']:.1f}GB")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader):
        """Validate model."""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(self.device, non_blocking=True), hr_imgs.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        sr_imgs = self.model(lr_imgs)
                else:
                    sr_imgs = self.model(lr_imgs)
                
                # Calculate PSNR
                mse = F.mse_loss(sr_imgs, hr_imgs)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()
                
                # Calculate SSIM (simplified)
                ssim = self._calculate_ssim(sr_imgs, hr_imgs)
                total_ssim += ssim.item()
        
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        
        return avg_psnr, avg_ssim
    
    def _calculate_ssim(self, sr_imgs, hr_imgs):
        """Calculate SSIM (simplified version)."""
        # This is a simplified SSIM calculation
        # For production, use proper SSIM implementation
        mu_x = sr_imgs.mean()
        mu_y = hr_imgs.mean()
        sigma_x = sr_imgs.var()
        sigma_y = hr_imgs.var()
        sigma_xy = ((sr_imgs - mu_x) * (hr_imgs - mu_y)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim
    
    def save_checkpoint(self, epoch: int, psnr: float, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'psnr': psnr,
            'stage': self.current_stage,
            'config': self.config
        }
        
        if hasattr(self, 'discriminator'):
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint and hasattr(self, 'discriminator'):
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
        return checkpoint['epoch'], checkpoint['psnr']
    
    def advance_stage(self):
        """Advance to next training stage."""
        # 清理当前阶段的GPU内存
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        torch.cuda.empty_cache()  # 强制清理GPU缓存
        
        self.current_stage += 1
        print(f"Advancing to stage {self.current_stage}")
        
        # Rebuild model for new stage
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 重新编译模型
        if self.compile_model and hasattr(torch, 'compile'):
            print("Recompiling model for new stage...")
            self.model = torch.compile(self.model, mode='max-autotune')
        
        # Rebuild optimizer with new learning rate
        self.config['lr'] *= self.config.get('stage_lr_decay', 0.5)
        self.optimizer = self._build_optimizer()
        
        # 再次清理缓存
        torch.cuda.empty_cache()

def train(config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader):
    """Main training function."""
    trainer = MultiStageTrainer(config)
    
    for stage in range(config.get('num_stages', 4)):
        # Set the current stage without advancing (since trainer starts at stage 0)
        if stage > 0:
            trainer.advance_stage()
        
        print(f"Training stage {trainer.current_stage}")
        
        for epoch in range(config['epochs_per_stage']):
            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % config.get('val_freq', 1) == 0:
                psnr, ssim = trainer.validate(val_loader)
                print(f"Epoch {epoch}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
                
                # Save best model
                if psnr > trainer.best_psnr:
                    trainer.best_psnr = psnr
                    trainer.save_checkpoint(
                        epoch, psnr, 
                        f"models/best_stage_{trainer.current_stage}.pth"
                    )
        
        # Save stage checkpoint
        trainer.save_checkpoint(
            config['epochs_per_stage'] - 1, trainer.best_psnr,
            f"models/stage_{trainer.current_stage}_final.pth"
        ) 