 #!/usr/bin/env python3
"""
多阶段模型评估脚本
支持评估stage 0-3的所有模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import our modules
from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock
from modules.transformer import LightweightTransformer
from modules.gan import GANSplice

class MultiStageModelBuilder:
    """构建多阶段训练的模型"""
    
    @staticmethod
    def build_model(config: Dict[str, Any], stage: int, device: torch.device):
        """根据阶段构建对应的模型"""
        base_channels = config['base_channels']
        num_blocks = config['num_blocks']
        scale_factor = config['scale_factor']
        
        # Stage 0: 基础backbone
        if stage == 0:
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=scale_factor
            )
        
        # Stage 1: 添加注意力机制
        elif stage == 1:
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=scale_factor
            )
            # 添加注意力块
            for i, block in enumerate(model.body):
                if i % 2 == 0:  # 每隔一个block添加注意力
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    model.body[i] = nn.Sequential(block, attention)
        
        # Stage 2: 添加Transformer
        elif stage == 2:
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=scale_factor
            )
            # 添加注意力块
            for i, block in enumerate(model.body):
                if i % 2 == 0:
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    model.body[i] = nn.Sequential(block, attention)
            
            # 添加Transformer
            transformer = LightweightTransformer(base_channels)
            mid_idx = len(model.body) // 2
            model.body.insert(mid_idx, transformer)
        
        # Stage 3: 添加GAN组件
        elif stage == 3:
            model = LightweightBackbone(
                in_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks,
                scale_factor=scale_factor
            )
            # 添加注意力块
            for i, block in enumerate(model.body):
                if i % 2 == 0:
                    attention = EfficientAttentionBlock(base_channels, 'omni')
                    model.body[i] = nn.Sequential(block, attention)
            
            # 添加Transformer
            transformer = LightweightTransformer(base_channels)
            mid_idx = len(model.body) // 2
            model.body.insert(mid_idx, transformer)
        
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        return model.to(device)

class EvalDataset(Dataset):
    """评估数据集"""
    def __init__(self, hr_dir: str, lr_dir: Optional[str] = None, scale_factor: int = 4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale_factor = scale_factor
        
        # 获取图像文件列表
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if lr_dir and os.path.exists(lr_dir):
            self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.lr_files = None
            
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # 加载HR图像
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        if self.lr_files:
            # 加载LR图像
            lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        else:
            # 生成LR图像
            h, w = hr_img.shape[:2]
            lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # 转换为tensor
        hr_tensor = torch.from_numpy(hr_img).float().permute(2, 0, 1) / 255.0
        lr_tensor = torch.from_numpy(lr_img).float().permute(2, 0, 1) / 255.0
        
        return lr_tensor, hr_tensor, self.hr_files[idx]

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_dataset(self, test_loader: DataLoader, save_results: bool = False, 
                        output_dir: str = "results") -> Dict[str, float]:
        """评估数据集"""
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for lr_imgs, hr_imgs, filenames in tqdm(test_loader, desc="Evaluating"):
                try:
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
                    # 生成超分辨率图像
                    sr_imgs = self.model(lr_imgs)
                    sr_imgs = torch.clamp(sr_imgs, 0, 1)
                    
                    # 计算指标
                    for i in range(lr_imgs.size(0)):
                        # PSNR
                        psnr_val = self._calculate_psnr(sr_imgs[i:i+1], hr_imgs[i:i+1])
                        total_psnr += psnr_val
                        
                        # SSIM
                        ssim_val = self._calculate_ssim(sr_imgs[i:i+1], hr_imgs[i:i+1])
                        total_ssim += ssim_val
                        
                        # 保存结果图像
                        if save_results:
                            self._save_result_images(
                                lr_imgs[i], sr_imgs[i], hr_imgs[i], 
                                filenames[i], output_dir
                            )
                    
                    num_samples += lr_imgs.size(0)
                    
                    # 清理GPU内存
                    del lr_imgs, hr_imgs, sr_imgs
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM at sample {num_samples}, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"Error processing batch at sample {num_samples}: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        return {
            'PSNR': float(total_psnr / num_samples) if num_samples > 0 else 0.0,
            'SSIM': float(total_ssim / num_samples) if num_samples > 0 else 0.0,
            'num_samples': int(num_samples)
        }
    
    def _calculate_psnr(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(sr_imgs, hr_imgs)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def _calculate_ssim(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """计算SSIM"""
        # 转换为numpy
        sr_np = sr_imgs.cpu().numpy().squeeze().transpose(1, 2, 0)
        hr_np = hr_imgs.cpu().numpy().squeeze().transpose(1, 2, 0)
        
        return ssim(sr_np, hr_np, multichannel=True, data_range=1.0, channel_axis=2)
    
    def _save_result_images(self, lr_img: torch.Tensor, sr_img: torch.Tensor, 
                           hr_img: torch.Tensor, filename: str, output_dir: str):
        """保存结果图像"""
        # 转换为numpy并反归一化
        lr_np = (lr_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sr_np = (sr_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        hr_np = (hr_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # 拼接图像 (LR | SR | HR)
        combined = np.hstack([lr_np, sr_np, hr_np])
        
        # 保存
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def load_model_from_checkpoint(model_path: str, device: torch.device):
    """从checkpoint加载模型"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    stage = checkpoint.get('stage', 0)
    
    print(f"Model stage: {stage}")
    print(f"Model config: base_channels={config['base_channels']}, "
          f"num_blocks={config['num_blocks']}, scale_factor={config['scale_factor']}")
    
    # 构建模型
    model = MultiStageModelBuilder.build_model(config, stage, device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, stage

def benchmark_model(model: nn.Module, device: torch.device, 
                   input_size: Tuple[int, int] = (128, 128), 
                   num_runs: int = 50) -> Dict[str, float]:
    """性能基准测试"""
    model.eval()
    
    # 创建测试输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 基准测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    # 内存使用
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_inference_time_ms': float(avg_time * 1000),
        'fps': float(fps),
        'memory_allocated_mb': float(memory_allocated),
        'memory_reserved_mb': float(memory_reserved)
    }

def compare_all_stages(model_dir: str, test_data_dir: str, device: torch.device):
    """比较所有阶段的模型性能"""
    results = {}
    
    # 查找所有阶段的模型文件
    stage_files = []
    for stage in range(4):
        stage_file = os.path.join(model_dir, f"628_stage_{stage}_final.pth")
        
        if os.path.exists(stage_file):
            stage_files.append((stage, stage_file))
    
    if not stage_files:
        print(f"No stage models found in {model_dir}")
        return results
    
    # 创建测试数据集
    test_dataset = EvalDataset(
        hr_dir=os.path.join(test_data_dir, 'HR'),
        lr_dir=os.path.join(test_data_dir, 'LR'),
        scale_factor=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1  # 减少worker数量避免内存问题
    )
    
    print(f"Found {len(test_dataset)} test images")
    
    # 为了避免内存问题，限制评估样本数量
    if len(test_dataset) > 1000:
        print(f"Limiting evaluation to first 1000 samples for memory efficiency")
        # 创建子集数据加载器
        subset_indices = list(range(1000))
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, subset_indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
    
    # 评估每个阶段
    for stage, model_path in stage_files:
        print(f"\n{'='*50}")
        print(f"Evaluating Stage {stage}")
        print(f"{'='*50}")
        
        try:
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 加载模型
            model, config, _ = load_model_from_checkpoint(model_path, device)
            
            # 评估数据集
            evaluator = ModelEvaluator(model, device)
            eval_results = evaluator.evaluate_dataset(test_loader)
            
            # 性能基准测试（使用较小的输入尺寸）
            if stage >= 2:  # 对于包含Transformer的模型使用更小的输入
                benchmark_results = benchmark_model(model, device, input_size=(64, 64))
            else:
                benchmark_results = benchmark_model(model, device)
            
            # 合并结果
            stage_results = {
                **eval_results,
                **benchmark_results,
                'model_path': model_path,
                'stage': stage
            }
            
            results[f"stage_{stage}"] = stage_results
            
            # 清理模型和GPU内存
            del model, evaluator
            torch.cuda.empty_cache()
            
            # 打印结果
            print(f"PSNR: {eval_results['PSNR']:.2f} dB")
            print(f"SSIM: {eval_results['SSIM']:.4f}")
            print(f"Inference Time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
            print(f"FPS: {benchmark_results['fps']:.1f}")
            print(f"GPU Memory: {benchmark_results['memory_allocated_mb']:.1f} MB")
            
        except Exception as e:
            print(f"Error evaluating stage {stage}: {e}")
            results[f"stage_{stage}"] = {'error': str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='多阶段模型评估工具')
    parser.add_argument('--model_path', type=str, help='单个模型文件路径')
    parser.add_argument('--model_dir', type=str, default='models', help='模型目录')
    parser.add_argument('--test_data_dir', type=str, default='data/val', help='测试数据目录')
    parser.add_argument('--compare_all', action='store_true', help='比较所有阶段模型')
    parser.add_argument('--save_results', action='store_true', help='保存结果图像')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='结果输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.compare_all:
        # 比较所有阶段模型
        print("Comparing all stage models...")
        results = compare_all_stages(args.model_dir, args.test_data_dir, device)
        
        # 保存比较结果
        import json
        with open('stage_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("STAGE COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Stage':<8} {'PSNR (dB)':<12} {'SSIM':<8} {'FPS':<8} {'Memory (MB)':<12}")
        print("-" * 60)
        
        for stage_name, result in results.items():
            if 'error' not in result:
                print(f"{stage_name:<8} {result['PSNR']:<12.2f} {result['SSIM']:<8.4f} "
                      f"{result['fps']:<8.1f} {result['memory_allocated_mb']:<12.1f}")
    
    elif args.model_path:
        # 评估单个模型
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        # 加载模型
        model, config, stage = load_model_from_checkpoint(args.model_path, device)
        
        # 创建测试数据集
        test_dataset = EvalDataset(
            hr_dir=os.path.join(args.test_data_dir, 'HR'),
            lr_dir=os.path.join(args.test_data_dir, 'LR'),
            scale_factor=config['scale_factor']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Evaluating {len(test_dataset)} test images...")
        
        # 评估
        evaluator = ModelEvaluator(model, device)
        results = evaluator.evaluate_dataset(
            test_loader, 
            save_results=args.save_results,
            output_dir=args.output_dir
        )
        
        # 性能基准测试
        benchmark_results = benchmark_model(model, device)
        
        # 打印结果
        print(f"\nEvaluation Results:")
        print(f"PSNR: {results['PSNR']:.2f} dB")
        print(f"SSIM: {results['SSIM']:.4f}")
        print(f"Samples: {results['num_samples']}")
        
        print(f"\nBenchmark Results:")
        print(f"Average Inference Time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
        print(f"FPS: {benchmark_results['fps']:.1f}")
        print(f"GPU Memory Allocated: {benchmark_results['memory_allocated_mb']:.1f} MB")
        
        if args.save_results:
            print(f"\nResults saved to: {args.output_dir}")
    
    else:
        print("Please specify --model_path for single model evaluation or --compare_all for stage comparison")

if __name__ == "__main__":
    main()