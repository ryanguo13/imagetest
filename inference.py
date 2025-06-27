#!/usr/bin/env python3
"""
简单的超分辨率推理脚本
支持对单张图像进行4x超分辨率处理
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# Import our modules
from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock  
from modules.transformer import LightweightTransformer

def build_model_for_stage(config, stage):
    """根据阶段构建模型"""
    base_channels = config['base_channels']
    num_blocks = config['num_blocks']
    scale_factor = config['scale_factor']
    
    # Stage 0: 基础backbone
    model = LightweightBackbone(
        in_channels=3,
        base_channels=base_channels,
        num_blocks=num_blocks,
        scale_factor=scale_factor
    )
    
    if stage >= 1:
        # Stage 1+: 添加注意力机制
        for i, block in enumerate(model.body):
            if i % 2 == 0:  # 每隔一个block添加注意力
                attention = EfficientAttentionBlock(base_channels, 'omni')
                model.body[i] = nn.Sequential(block, attention)
    
    if stage >= 2:
        # Stage 2+: 添加Transformer
        transformer = LightweightTransformer(base_channels)
        mid_idx = len(model.body) // 2
        model.body.insert(mid_idx, transformer)
    
    # Stage 3不需要额外处理，因为GAN部分只在训练时使用
    
    return model

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    stage = checkpoint.get('stage', 0)
    
    print(f"Model stage: {stage}")
    print(f"Config: base_channels={config['base_channels']}, num_blocks={config['num_blocks']}")
    
    # 构建模型
    model = build_model_for_stage(config, stage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def preprocess_image(image_path, device):
    """预处理输入图像"""
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为tensor并归一化
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加batch维度
    
    return img_tensor, img

def postprocess_image(tensor):
    """后处理输出图像"""
    # 去除batch维度并限制到[0,1]
    tensor = torch.clamp(tensor.squeeze(0), 0, 1)
    
    # 转换为numpy
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    
    return img

def super_resolve_image(model, image_path, output_path, device):
    """对单张图像进行超分辨率处理"""
    
    # 预处理
    lr_tensor, lr_img = preprocess_image(image_path, device)
    
    print(f"Input image shape: {lr_img.shape}")
    
    # 推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # 后处理
    sr_img = postprocess_image(sr_tensor)
    
    print(f"Output image shape: {sr_img.shape}")
    
    # 保存结果
    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sr_img_bgr)
    
    print(f"Super-resolved image saved to: {output_path}")
    
    # 创建对比图
    comparison_path = output_path.replace('.', '_comparison.')
    create_comparison_image(lr_img, sr_img, comparison_path)
    
    return sr_img

def create_comparison_image(lr_img, sr_img, output_path):
    """创建LR vs SR对比图"""
    
    # 调整LR图像大小以匹配SR图像
    lr_upscaled = cv2.resize(lr_img, (sr_img.shape[1], sr_img.shape[0]), 
                            interpolation=cv2.INTER_CUBIC)
    
    # 水平拼接
    comparison = np.hstack([lr_upscaled, sr_img])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Bicubic', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Super-Resolution', (sr_img.shape[1] + 10, 30), 
                font, 1, (255, 255, 255), 2)
    
    # 保存
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, comparison_bgr)
    
    print(f"Comparison image saved to: {output_path}")

def batch_inference(model, input_dir, output_dir, device):
    """批量推理目录中的所有图像"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        output_path = os.path.join(output_dir, f"sr_{image_path.name}")
        
        try:
            super_resolve_image(model, str(image_path), output_path, device)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

def benchmark_inference_speed(model, device, image_size=(128, 128), num_runs=50):
    """测试推理速度"""
    
    print(f"Benchmarking inference speed with {image_size} input...")
    
    # 创建随机输入
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.1f}")
    
    return avg_time, fps

def main():
    parser = argparse.ArgumentParser(description='超分辨率推理工具')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, help='输入图像路径')
    parser.add_argument('--output', type=str, help='输出图像路径')
    parser.add_argument('--input_dir', type=str, help='输入图像目录（批量处理）')
    parser.add_argument('--output_dir', type=str, help='输出图像目录（批量处理）')
    parser.add_argument('--benchmark', action='store_true', help='运行速度基准测试')
    
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # 加载模型
    try:
        model, config = load_model(args.model, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 速度基准测试
    if args.benchmark:
        benchmark_inference_speed(model, device)
    
    # 单张图像推理
    if args.input and args.output:
        if not os.path.exists(args.input):
            print(f"Input image not found: {args.input}")
            return
        
        try:
            super_resolve_image(model, args.input, args.output, device)
        except Exception as e:
            print(f"Error during inference: {e}")
    
    # 批量推理
    elif args.input_dir and args.output_dir:
        if not os.path.exists(args.input_dir):
            print(f"Input directory not found: {args.input_dir}")
            return
        
        try:
            batch_inference(model, args.input_dir, args.output_dir, device)
        except Exception as e:
            print(f"Error during batch inference: {e}")
    
    # 如果没有指定推理任务，只运行基准测试
    elif not args.benchmark:
        print("Please specify --input/--output for single image or --input_dir/--output_dir for batch processing")
        print("Or use --benchmark to test inference speed")

if __name__ == "__main__":
    main() 