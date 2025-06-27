#!/usr/bin/env python3
"""
GPU优化脚本 - 一键优化你的训练配置以榨干GPU性能

使用方法:
    python optimize_gpu.py --config configs/lightweight_config.yaml
    
功能:
    1. 自动检测GPU规格
    2. 生成优化配置
    3. 可选：自动搜索最优batch size
    4. 实时GPU监控
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from utils.gpu_optimizer import auto_optimize_config, GPUMonitor, BatchSizeOptimizer
    from src.main import create_model
    GPU_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPU optimizer import failed: {e}")
    print("Will provide basic optimization without advanced monitoring.")
    GPU_OPTIMIZER_AVAILABLE = False

def print_gpu_info():
    """打印GPU信息"""
    print("🚀 GPU优化工具")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA设备")
        print("建议: 请确保安装了正确的PyTorch GPU版本")
        return False
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"⚡ PyTorch版本: {torch.__version__}")
    print()
    
    return True

def create_basic_optimized_config(base_config_path: str, output_path: str):
    """创建基础优化配置（不依赖GPU检测）"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 基础优化：不依赖GPU检测
    config.update({
        'batch_size': max(config.get('batch_size', 16) * 2, 32),  # 至少翻倍
        'base_channels': max(config.get('base_channels', 32) * 2, 64),  # 至少翻倍
        'num_blocks': max(config.get('num_blocks', 4) * 2, 8),  # 至少翻倍
        'num_workers': 8,  # 增加数据加载并行度
        'patch_size': max(config.get('patch_size', 128), 192),  # 增大patch
        'use_mixed_precision': True,
        'use_perceptual': True,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
        'grad_clip': 1.0,
    })
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path

def optimize_config(base_config_path: str, output_path: str = None):
    """优化配置文件"""
    if output_path is None:
        base_name = Path(base_config_path).stem
        output_path = f"configs/{base_name}_gpu_optimized.yaml"
    
    print("🔧 正在优化配置...")
    
    if GPU_OPTIMIZER_AVAILABLE:
        optimized_path = auto_optimize_config(base_config_path, output_path)
    else:
        print("使用基础优化模式（无GPU检测）...")
        optimized_path = create_basic_optimized_config(base_config_path, output_path)
    
    # 显示优化前后对比
    with open(base_config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    with open(optimized_path, 'r') as f:
        optimized_config = yaml.safe_load(f)
    
    print("\n📈 优化对比:")
    print("-" * 50)
    improvements = [
        ('batch_size', 'Batch Size'),
        ('base_channels', 'Base Channels'), 
        ('num_blocks', 'Num Blocks'),
        ('num_workers', 'Num Workers'),
        ('patch_size', 'Patch Size'),
    ]
    
    for key, name in improvements:
        old_val = original_config.get(key, 'N/A')
        new_val = optimized_config.get(key, 'N/A')
        if old_val != new_val:
            print(f"{name:15}: {old_val:>6} → {new_val:>6} {'🚀' if new_val > old_val else '💡'}")
    
    # 新增的优化功能
    new_features = [
        ('use_mixed_precision', '混合精度训练'),
        ('pin_memory', '内存固定'),
        ('persistent_workers', '持久化Worker'),
        ('compile_model', '模型编译优化'),
    ]
    
    print("\n✨ 新增优化功能:")
    print("-" * 50)
    for key, name in new_features:
        if optimized_config.get(key, False):
            print(f"✅ {name}")
    
    return optimized_path

def benchmark_performance(config_path: str):
    """性能基准测试"""
    if not GPU_OPTIMIZER_AVAILABLE:
        print("\n⚠️ 基准测试需要完整的GPU优化器，当前不可用")
        print("建议手动调整batch_size: 32 → 48 → 64，直到显存不够为止")
        return config_path
    
    print("\n🏃 运行性能基准测试...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型用于测试
    model = create_model(config, stage=0)
    
    # 创建样本数据
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    sample_data = torch.randn(batch_size * 2, 3, patch_size, patch_size)
    
    # 自动搜索最优batch size
    optimizer = BatchSizeOptimizer(model, config)
    optimal_batch_size, stats = optimizer.find_optimal_batch_size(
        sample_data, target_gpu_util=85.0, max_batch_size=batch_size * 4
    )
    
    # 更新配置文件
    if optimal_batch_size != batch_size:
        print(f"\n📝 建议更新batch_size: {batch_size} → {optimal_batch_size}")
        config['batch_size'] = optimal_batch_size
        
        # 保存更新的配置
        output_path = config_path.replace('.yaml', '_optimized.yaml')
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 已保存优化配置到: {output_path}")
        return output_path
    
    return config_path

def monitor_training(duration: int = 60):
    """实时监控训练过程中的GPU使用情况"""
    if not torch.cuda.is_available():
        print("\n❌ 未检测到CUDA设备，无法监控GPU")
        return
    
    if not GPU_OPTIMIZER_AVAILABLE:
        print(f"\n👀 启动基础GPU监控 ({duration}秒)...")
        print("提示: 现在开始你的训练，我会显示基础GPU信息")
        print("按 Ctrl+C 提前结束监控")
        
        try:
            import time
            start_time = time.time()
            while time.time() - start_time < duration:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_util = (memory_allocated / memory_total) * 100
                
                print(f"\r🎯 显存使用: {memory_util:>3.0f}% ({memory_allocated:.1f}GB / {memory_total:.1f}GB)", end='')
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        print("\n\n💡 提示: 如果显存使用率低，可以增大batch_size")
        return
    
    print(f"\n👀 启动GPU监控 ({duration}秒)...")
    print("提示: 现在开始你的训练，我会监控GPU使用情况")
    print("按 Ctrl+C 提前结束监控")
    
    monitor = GPUMonitor()
    monitor.start_monitoring(interval=1.0)
    
    try:
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            stats = monitor.get_current_stats()
            if stats:
                gpu_util = stats.get('gpu_utilization', 0)
                memory_util = stats.get('memory_utilization', 0)
                memory_used = stats.get('memory_allocated_gb', 0)
                
                print(f"\r🎯 GPU利用率: {gpu_util:>3.0f}% | "
                      f"显存使用: {memory_util:>3.0f}% ({memory_used:.1f}GB)", end='')
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()
        
        # 显示总结
        avg_stats = monitor.get_average_stats()
        if avg_stats:
            print(f"\n\n📊 监控总结:")
            print(f"平均GPU利用率: {avg_stats.get('avg_gpu_util', 0):.1f}%")
            print(f"平均显存使用: {avg_stats.get('avg_memory_util', 0):.1f}%")
            print(f"峰值显存使用: {avg_stats.get('peak_memory_gb', 0):.1f}GB")
            
            avg_gpu_util = avg_stats.get('avg_gpu_util', 0)
            if avg_gpu_util < 70:
                print("💡 建议: GPU利用率较低，可以考虑增大batch_size或模型复杂度")
            elif avg_gpu_util > 95:
                print("⚠️  警告: GPU利用率过高，可能影响系统稳定性")
            else:
                print("✅ GPU利用率良好!")

def main():
    parser = argparse.ArgumentParser(description='GPU优化工具 - 榨干你的显卡性能!')
    parser.add_argument('--config', type=str, default='configs/lightweight_config.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出配置文件路径')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--monitor', type=int, default=0,
                       help='监控GPU使用情况(秒数，0表示不监控)')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='跳过配置优化，仅运行其他功能')
    
    args = parser.parse_args()
    
    # 检查GPU
    if not print_gpu_info():
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("可用的配置文件:")
        config_dir = Path("configs")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        return
    
    optimized_config_path = args.config
    
    # 优化配置
    if not args.skip_optimization:
        optimized_config_path = optimize_config(args.config, args.output)
        print(f"\n🎉 配置优化完成: {optimized_config_path}")
    
    # 性能基准测试
    if args.benchmark:
        try:
            optimized_config_path = benchmark_performance(optimized_config_path)
        except Exception as e:
            print(f"⚠️  基准测试失败: {e}")
    
    # GPU监控
    if args.monitor > 0:
        monitor_training(args.monitor)
    
    print(f"\n🚀 使用优化后的配置训练:")
    print(f"python src/main.py --config {optimized_config_path}")
    
    print(f"\n💡 其他建议:")
    print(f"1. 训练时运行 'python optimize_gpu.py --monitor 300' 来监控GPU使用情况")
    print(f"2. 如果仍然利用率不高，尝试增大 batch_size 或 base_channels")
    print(f"3. 确保数据加载不是瓶颈：增大 num_workers")

if __name__ == "__main__":
    main() 