#!/usr/bin/env python3
"""
快速评估脚本 - 内存友好版本
只评估100张图像，避免内存问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from tqdm import tqdm
import json
from pathlib import Path

from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock
from modules.transformer import LightweightTransformer

def build_model_for_stage(config, stage):
    """根据阶段构建模型"""
    base_channels = config['base_channels']
    num_blocks = config['num_blocks']
    scale_factor = config['scale_factor']
    
    model = LightweightBackbone(
        in_channels=3,
        base_channels=base_channels,
        num_blocks=num_blocks,
        scale_factor=scale_factor
    )
    
    if stage >= 1:
        for i, block in enumerate(model.body):
            if i % 2 == 0:
                attention = EfficientAttentionBlock(base_channels, 'omni')
                model.body[i] = nn.Sequential(block, attention)
    
    if stage >= 2:
        transformer = LightweightTransformer(base_channels)
        mid_idx = len(model.body) // 2
        model.body.insert(mid_idx, transformer)
    
    return model

def load_model_from_checkpoint(model_path, device):
    """从checkpoint加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    stage = checkpoint.get('stage', 0)
    
    model = build_model_for_stage(config, stage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, stage

def calculate_psnr(sr_img, hr_img):
    """计算PSNR"""
    mse = F.mse_loss(sr_img, hr_img)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim_simple(sr_img, hr_img):
    """简化的SSIM计算"""
    sr_np = sr_img.cpu().numpy().squeeze().transpose(1, 2, 0)
    hr_np = hr_img.cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # 简化版SSIM
    mu_sr = sr_np.mean()
    mu_hr = hr_np.mean()
    sigma_sr = sr_np.var()
    sigma_hr = hr_np.var()
    sigma_sr_hr = ((sr_np - mu_sr) * (hr_np - mu_hr)).mean()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_sr * mu_hr + c1) * (2 * sigma_sr_hr + c2)) / \
           ((mu_sr ** 2 + mu_hr ** 2 + c1) * (sigma_sr + sigma_hr + c2))
    
    return float(ssim)

def evaluate_model_quick(model_path, data_dir, device, num_samples=100):
    """快速评估模型"""
    print(f"Quick evaluation of {model_path}")
    
    # 加载模型
    model, config, stage = load_model_from_checkpoint(model_path, device)
    
    # 获取测试图像
    hr_dir = os.path.join(data_dir, 'HR')
    lr_dir = os.path.join(data_dir, 'LR')
    
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith('.png')])[:num_samples]
    
    total_psnr = 0.0
    total_ssim = 0.0
    successful_samples = 0
    
    print(f"Evaluating {len(hr_files)} samples...")
    
    with torch.no_grad():
        for i, filename in enumerate(tqdm(hr_files)):
            try:
                # 加载图像
                hr_path = os.path.join(hr_dir, filename)
                lr_path = os.path.join(lr_dir, filename)
                
                hr_img = cv2.imread(hr_path)
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                hr_tensor = torch.from_numpy(hr_img).float().permute(2, 0, 1) / 255.0
                hr_tensor = hr_tensor.unsqueeze(0).to(device)
                
                if os.path.exists(lr_path):
                    lr_img = cv2.imread(lr_path)
                    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                else:
                    # 生成LR图像
                    h, w = hr_img.shape[:2]
                    lr_h, lr_w = h // 4, w // 4
                    lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                
                lr_tensor = torch.from_numpy(lr_img).float().permute(2, 0, 1) / 255.0
                lr_tensor = lr_tensor.unsqueeze(0).to(device)
                
                # 推理
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
                
                # 计算指标
                psnr_val = calculate_psnr(sr_tensor, hr_tensor)
                ssim_val = calculate_ssim_simple(sr_tensor, hr_tensor)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                successful_samples += 1
                
                # 清理内存
                del lr_tensor, hr_tensor, sr_tensor
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                torch.cuda.empty_cache()
                continue
    
    if successful_samples > 0:
        avg_psnr = total_psnr / successful_samples
        avg_ssim = total_ssim / successful_samples
    else:
        avg_psnr = 0.0
        avg_ssim = 0.0
    
    # 清理模型
    del model
    torch.cuda.empty_cache()
    
    return {
        'stage': stage,
        'PSNR': float(avg_psnr),
        'SSIM': float(avg_ssim),
        'successful_samples': successful_samples,
        'model_path': model_path
    }

def benchmark_quick(model_path, device):
    """快速基准测试"""
    try:
        model, config, stage = load_model_from_checkpoint(model_path, device)
        
        # 选择合适的输入尺寸
        if stage >= 2:
            input_size = (64, 64)
        else:
            input_size = (128, 128)
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # 计时
        import time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        num_runs = 20
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        # 清理
        del model, dummy_input
        torch.cuda.empty_cache()
        
        return {
            'avg_inference_time_ms': float(avg_time * 1000),
            'fps': float(fps),
            'input_size': input_size
        }
        
    except Exception as e:
        print(f"Benchmark failed for {model_path}: {e}")
        torch.cuda.empty_cache()
        return {'error': str(e)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_dir = "models"
    data_dir = "data/val"
    
    # 查找所有模型文件
    stage_files = []
    for stage in range(4):
        stage_file = os.path.join(model_dir, f"stage_{stage}_final.pth")
        if os.path.exists(stage_file):
            stage_files.append((stage, stage_file))
    
    if not stage_files:
        print(f"No stage models found in {model_dir}")
        return
    
    print(f"Found {len(stage_files)} stage models")
    
    results = {}
    
    for stage, model_path in stage_files:
        print(f"\n{'='*50}")
        print(f"Evaluating Stage {stage}")
        print(f"{'='*50}")
        
        # 评估
        eval_results = evaluate_model_quick(model_path, data_dir, device, num_samples=100)
        
        # 基准测试
        benchmark_results = benchmark_quick(model_path, device)
        
        # 合并结果
        stage_results = {**eval_results, **benchmark_results}
        results[f"stage_{stage}"] = stage_results
        
        # 打印结果
        if 'error' not in eval_results:
            print(f"PSNR: {eval_results['PSNR']:.2f} dB")
            print(f"SSIM: {eval_results['SSIM']:.4f}")
            print(f"Samples: {eval_results['successful_samples']}/100")
        
        if 'error' not in benchmark_results:
            print(f"Inference Time: {benchmark_results['avg_inference_time_ms']:.2f} ms")
            print(f"FPS: {benchmark_results['fps']:.1f}")
    
    # 保存结果
    with open('quick_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("QUICK EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Stage':<8} {'PSNR (dB)':<12} {'SSIM':<8} {'FPS':<8} {'Samples':<8}")
    print("-" * 60)
    
    for stage_name, result in results.items():
        if 'error' not in result:
            print(f"{stage_name:<8} {result.get('PSNR', 0):<12.2f} {result.get('SSIM', 0):<8.4f} "
                  f"{result.get('fps', 0):<8.1f} {result.get('successful_samples', 0):<8}")
    
    print("\nResults saved to quick_eval_results.json")

if __name__ == "__main__":
    main() 