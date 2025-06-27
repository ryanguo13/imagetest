#!/usr/bin/env python3
"""
Main entry point for the lightweight super-resolution project.
Follows the roadmap steps:
1. Model architecture (PixelShuffle, BSConv, etc.)
2. Efficiency modules (Omni-SR attention, ConvNeXt, etc.)
3. Training acceleration (LoRA, distillation)
4. Optional enhancements (Transformer, GAN, etc.)
5. Training and evaluation pipeline
"""

import argparse
import os
import sys
import torch
from typing import Dict, Any
from dataclasses import asdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock, OmniSRAttention
from modules.lora import ConvLoRA, DistillationLoss
from modules.transformer import ProgressiveFocusTransformer, LightweightTransformer
from modules.gan import GANSplice, AdversarialLoss, PerceptualLoss
from training.train import train, MultiStageTrainer
from training.evaluate import evaluate_model, benchmark_model, SuperResolutionDataset
from utils.config import ConfigManager, WarmStartManager, ExperimentTracker, DEFAULT_CONFIG

def create_model(config: Dict[str, Any], stage: int = 0):
    """
    Create model based on current stage.
    """
    base_channels = config['base_channels']
    num_blocks = config['num_blocks']
    scale_factor = config['scale_factor']
    
    # Stage 0: Basic backbone
    if stage == 0:
        return LightweightBackbone(
            in_channels=3,
            base_channels=base_channels,
            num_blocks=num_blocks,
            scale_factor=scale_factor
        )
    
    # Stage 1: Add attention
    elif stage == 1:
        model = LightweightBackbone(
            in_channels=3,
            base_channels=base_channels,
            num_blocks=num_blocks,
            scale_factor=scale_factor
        )
        # Add attention blocks
        for i, block in enumerate(model.body):
            if i % 2 == 0:  # Add attention every other block
                attention = OmniSRAttention(base_channels)
                # Insert attention after ConvNeXt block
                model.body[i] = torch.nn.Sequential(block, attention)
        return model
    
    # Stage 2: Add transformer
    elif stage == 2:
        model = create_model(config, stage=1)  # Get stage 1 model
        # Add transformer module
        transformer = LightweightTransformer(base_channels)
        # Insert transformer in the middle
        mid_idx = len(model.body) // 2
        model.body.insert(mid_idx, transformer)
        return model
    
    # Stage 3: Add GAN components
    elif stage == 3:
        model = create_model(config, stage=2)  # Get stage 2 model
        return model

def setup_data_loaders(config: Dict[str, Any], data_dir: str):
    """
    Setup training and validation data loaders with GPU optimizations.
    """
    # Training dataset
    train_dataset = SuperResolutionDataset(
        hr_dir=os.path.join(data_dir, 'train', 'HR'),
        lr_dir=os.path.join(data_dir, 'train', 'LR'),
        scale_factor=config['scale_factor'],
        patch_size=config['patch_size'],
        is_train=True
    )
    
    # Validation dataset
    val_dataset = SuperResolutionDataset(
        hr_dir=os.path.join(data_dir, 'val', 'HR'),
        lr_dir=os.path.join(data_dir, 'val', 'LR'),
        scale_factor=config['scale_factor'],
        is_train=False
    )
    
    # GPU优化的数据加载器参数
    pin_memory = config.get('pin_memory', True) and torch.cuda.is_available()
    persistent_workers = config.get('persistent_workers', False) and config['num_workers'] > 0
    prefetch_factor = config.get('prefetch_factor', 2) if config['num_workers'] > 0 else 2
    
    # Data loaders with optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True  # 保持batch size一致，提高训练稳定性
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 4),  # 验证时也可以用更大的batch size
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    print(f"Data loader settings:")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Num workers: {config['num_workers']}")
    print(f"  - Pin memory: {pin_memory}")
    print(f"  - Persistent workers: {persistent_workers}")
    print(f"  - Prefetch factor: {prefetch_factor}")
    
    return train_loader, val_loader

def train_model(config_path: str, data_dir: str, experiment_name: str):
    """
    Train the model using multi-stage approach.
    """
    print(f"Starting training with config: {config_path}")
    print(f"Data directory: {data_dir}")
    print(f"Experiment name: {experiment_name}")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Setup experiment tracking
    tracker = ExperimentTracker(experiment_name, config_manager)
    
    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(asdict(config), data_dir)
    
    # Start training
    print("Starting multi-stage training...")
    train(asdict(config), train_loader, val_loader)
    
    print("Training completed!")

def evaluate_model_wrapper(model_path: str, test_data_dir: str):
    """
    Evaluate a trained model.
    """
    print(f"Evaluating model: {model_path}")
    print(f"Test data directory: {test_data_dir}")
    
    results = evaluate_model(model_path, test_data_dir)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    return results

def benchmark_model_wrapper(model_path: str):
    """
    Benchmark model performance.
    """
    print(f"Benchmarking model: {model_path}")
    
    results = benchmark_model(model_path)
    
    print("Benchmark Results:")
    for metric, value in results.items():
        if 'time' in metric:
            print(f"  {metric}: {value:.4f} seconds")
        elif 'fps' in metric:
            print(f"  {metric}: {value:.2f}")
        elif 'memory' in metric:
            print(f"  {metric}: {value:.2f} MB")
        else:
            print(f"  {metric}: {value}")
    
    return results

def create_default_configs():
    """
    Create default configuration files.
    """
    print("Creating default configuration files...")
    
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    
    print("Default configurations created in 'configs/' directory")

def main():
    parser = argparse.ArgumentParser(description='Lightweight Super-Resolution Training')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'benchmark', 'create_configs'],
                       help='Mode to run')
    parser.add_argument('--config', type=str, default='configs/balanced_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (for evaluate/benchmark)')
    parser.add_argument('--test_data_dir', type=str, default='data/test',
                       help='Path to test data directory (for evaluate)')
    parser.add_argument('--experiment_name', type=str, default='experiment_1',
                       help='Name for the experiment (for train)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    if args.mode == 'train':
        if not os.path.exists(args.config):
            print(f"Config file {args.config} not found. Creating default configs...")
            create_default_configs()
            print(f"Please use one of the created configs or specify a valid config path.")
            return
        
        train_model(args.config, args.data_dir, args.experiment_name)
    
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Please specify --model_path for evaluation")
            return
        
        evaluate_model_wrapper(args.model_path, args.test_data_dir)
    
    elif args.mode == 'benchmark':
        if not args.model_path:
            print("Please specify --model_path for benchmarking")
            return
        
        benchmark_model_wrapper(args.model_path)
    
    elif args.mode == 'create_configs':
        create_default_configs()

if __name__ == "__main__":
    main() 