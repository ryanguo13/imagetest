"""
GPU优化工具集
包括性能监控、自动batch size搜索、GPU利用率分析等功能
"""

import torch
import psutil
import time
import subprocess
import threading
from typing import Dict, Any, Optional, Tuple
import yaml
import os

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except (ImportError, Exception) as e:
    NVIDIA_ML_AVAILABLE = False
    print(f"Warning: NVIDIA-ML library not available ({type(e).__name__})")
    print("GPU monitoring will use basic PyTorch functions only.")

class GPUMonitor:
    """实时GPU性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'gpu_util': [],
            'memory_util': [],
            'temperature': [],
            'power_usage': [],
            'memory_used': [],
            'memory_total': 0
        }
        
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            
            if NVIDIA_ML_AVAILABLE:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.stats['memory_total'] = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / 1024**3
            else:
                self.handle = None
                self.stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            self.device_count = 0
            self.device_name = "No CUDA device"
            self.handle = None
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        if not torch.cuda.is_available():
            print("No CUDA device available for monitoring")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started GPU monitoring for {self.device_name}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("Stopped GPU monitoring")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                # PyTorch GPU stats
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                
                self.stats['memory_used'].append(memory_allocated)
                
                # NVIDIA-ML stats (if available)
                if NVIDIA_ML_AVAILABLE and self.handle:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
                        
                        self.stats['gpu_util'].append(util.gpu)
                        self.stats['memory_util'].append(memory.used / memory.total * 100)
                        self.stats['temperature'].append(temp)
                        self.stats['power_usage'].append(power)
                    except Exception as e:
                        # Fallback to basic stats
                        self.stats['gpu_util'].append(0)
                        self.stats['memory_util'].append(memory_allocated / self.stats['memory_total'] * 100)
                        self.stats['temperature'].append(0)
                        self.stats['power_usage'].append(0)
                else:
                    # Use basic PyTorch stats when NVIDIA-ML is not available
                    memory_util = (memory_allocated / self.stats['memory_total'] * 100) if self.stats['memory_total'] > 0 else 0
                    # Estimate GPU utilization based on memory usage (rough approximation)
                    estimated_gpu_util = min(memory_util * 1.2, 100)
                    
                    self.stats['gpu_util'].append(estimated_gpu_util)
                    self.stats['memory_util'].append(memory_util)
                    self.stats['temperature'].append(0)  # Not available without NVIDIA-ML
                    self.stats['power_usage'].append(0)  # Not available without NVIDIA-ML
                
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def get_current_stats(self) -> Dict[str, float]:
        """获取当前GPU统计信息"""
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'memory_total_gb': self.stats['memory_total'],
        }
        
        if self.stats['gpu_util']:
            stats.update({
                'gpu_utilization': self.stats['gpu_util'][-1] if self.stats['gpu_util'] else 0,
                'memory_utilization': self.stats['memory_util'][-1] if self.stats['memory_util'] else 0,
                'temperature': self.stats['temperature'][-1] if self.stats['temperature'] else 0,
                'power_usage': self.stats['power_usage'][-1] if self.stats['power_usage'] else 0,
            })
        
        return stats
    
    def get_average_stats(self, last_n: int = 100) -> Dict[str, float]:
        """获取平均统计信息"""
        if not self.stats['gpu_util']:
            return self.get_current_stats()
        
        n = min(last_n, len(self.stats['gpu_util']))
        if n == 0:
            return self.get_current_stats()
        
        return {
            'avg_gpu_util': sum(self.stats['gpu_util'][-n:]) / n,
            'avg_memory_util': sum(self.stats['memory_util'][-n:]) / n,
            'avg_temperature': sum(self.stats['temperature'][-n:]) / n,
            'avg_power_usage': sum(self.stats['power_usage'][-n:]) / n,
            'peak_memory_gb': max(self.stats['memory_used'][-n:]) if self.stats['memory_used'] else 0,
        }

class BatchSizeOptimizer:
    """自动搜索最优batch size"""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.monitor = GPUMonitor()
    
    def find_optimal_batch_size(self, sample_data: torch.Tensor, 
                               target_gpu_util: float = 85.0,
                               max_batch_size: int = 128,
                               min_batch_size: int = 1) -> Tuple[int, Dict[str, Any]]:
        """
        自动搜索最优batch size
        
        Args:
            sample_data: 样本数据用于测试
            target_gpu_util: 目标GPU利用率 (%)
            max_batch_size: 最大batch size
            min_batch_size: 最小batch size
            
        Returns:
            (optimal_batch_size, performance_stats)
        """
        print(f"Searching optimal batch size (target GPU util: {target_gpu_util}%)")
        
        if not torch.cuda.is_available():
            print("No CUDA device available, using batch_size=1")
            return 1, {}
        
        self.model.to(self.device)
        self.model.train()
        
        # 二分搜索最大可用batch size
        max_feasible_batch_size = self._find_max_batch_size(sample_data, max_batch_size)
        print(f"Maximum feasible batch size: {max_feasible_batch_size}")
        
        # 在可行范围内搜索最优batch size
        optimal_batch_size = self._search_optimal_batch_size(
            sample_data, min_batch_size, max_feasible_batch_size, target_gpu_util
        )
        
        # 获取最终性能统计
        performance_stats = self._benchmark_batch_size(sample_data, optimal_batch_size)
        
        print(f"Optimal batch size: {optimal_batch_size}")
        print(f"Performance stats: {performance_stats}")
        
        return optimal_batch_size, performance_stats
    
    def _find_max_batch_size(self, sample_data: torch.Tensor, max_batch_size: int) -> int:
        """二分搜索找到最大可行batch size"""
        left, right = 1, max_batch_size
        max_feasible = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # 测试是否可以运行
                batch_data = sample_data[:mid].to(self.device)
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    output = self.model(batch_data)
                    loss = torch.nn.functional.mse_loss(output, output)  # 虚拟loss
                
                # 如果成功，尝试更大的batch size
                max_feasible = mid
                left = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM，尝试更小的batch size
                    right = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        return max_feasible
    
    def _search_optimal_batch_size(self, sample_data: torch.Tensor, 
                                 min_batch_size: int, max_batch_size: int,
                                 target_gpu_util: float) -> int:
        """搜索最优batch size以达到目标GPU利用率"""
        best_batch_size = min_batch_size
        best_score = 0
        
        # 测试不同的batch size
        test_sizes = []
        step = max(1, (max_batch_size - min_batch_size) // 8)
        for bs in range(min_batch_size, max_batch_size + 1, step):
            test_sizes.append(bs)
        if max_batch_size not in test_sizes:
            test_sizes.append(max_batch_size)
        
        for batch_size in test_sizes:
            try:
                stats = self._benchmark_batch_size(sample_data, batch_size)
                
                # 计算分数：接近目标利用率且吞吐量高
                gpu_util = stats.get('avg_gpu_util', 0)
                throughput = stats.get('samples_per_sec', 0)
                
                # 利用率分数：距离目标越近越好
                util_score = 100 - abs(gpu_util - target_gpu_util)
                # 吞吐量分数
                throughput_score = throughput
                
                # 综合分数
                total_score = util_score * 0.7 + min(throughput_score / 10, 30) * 0.3
                
                print(f"Batch size {batch_size}: GPU util {gpu_util:.1f}%, "
                      f"throughput {throughput:.1f} samples/sec, score {total_score:.1f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_batch_size = batch_size
                    
            except Exception as e:
                print(f"Failed to test batch size {batch_size}: {e}")
                continue
        
        return best_batch_size
    
    def _benchmark_batch_size(self, sample_data: torch.Tensor, batch_size: int) -> Dict[str, float]:
        """基准测试特定batch size的性能"""
        torch.cuda.empty_cache()
        
        # 准备数据
        batch_data = sample_data[:batch_size].to(self.device)
        
        # 开始监控
        self.monitor.start_monitoring(interval=0.1)
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                output = self.model(batch_data)
        
        torch.cuda.synchronize()
        
        # 实际测试
        start_time = time.time()
        num_iterations = 20
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = self.model(batch_data)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 停止监控
        time.sleep(0.5)  # 让监控收集一些数据
        self.monitor.stop_monitoring()
        
        # 计算性能指标
        total_time = end_time - start_time
        samples_per_sec = (batch_size * num_iterations) / total_time
        
        # 获取监控统计
        monitor_stats = self.monitor.get_average_stats()
        
        return {
            'batch_size': batch_size,
            'samples_per_sec': samples_per_sec,
            'time_per_batch': total_time / num_iterations,
            **monitor_stats
        }

class ConfigOptimizer:
    """配置优化器"""
    
    def __init__(self, base_config_path: str):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def create_optimized_config(self, gpu_memory_gb: float, 
                               target_gpu_util: float = 85.0) -> Dict[str, Any]:
        """根据GPU内存创建优化配置"""
        config = self.base_config.copy()
        
        # 根据GPU内存调整参数
        if gpu_memory_gb >= 24:  # 高端GPU
            config.update({
                'batch_size': 64,
                'base_channels': 128,
                'num_blocks': 12,
                'patch_size': 256,
                'num_workers': 12,
                'use_mixed_precision': True,
                'use_perceptual': True,
            })
        elif gpu_memory_gb >= 12:  # 中端GPU
            config.update({
                'batch_size': 32,
                'base_channels': 96,
                'num_blocks': 8,
                'patch_size': 192,
                'num_workers': 8,
                'use_mixed_precision': True,
                'use_perceptual': True,
            })
        elif gpu_memory_gb >= 8:  # 入门级GPU
            config.update({
                'batch_size': 16,
                'base_channels': 64,
                'num_blocks': 6,
                'patch_size': 128,
                'num_workers': 6,
                'use_mixed_precision': True,
                'use_perceptual': False,
            })
        else:  # 低端GPU
            config.update({
                'batch_size': 8,
                'base_channels': 32,
                'num_blocks': 4,
                'patch_size': 96,
                'num_workers': 4,
                'use_mixed_precision': True,
                'use_perceptual': False,
            })
        
        # 启用所有GPU优化
        config.update({
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'compile_model': True,
        })
        
        return config
    
    def save_optimized_config(self, config: Dict[str, Any], output_path: str):
        """保存优化配置"""
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Optimized config saved to: {output_path}")

def auto_optimize_config(base_config_path: str = 'configs/lightweight_config.yaml',
                        output_path: str = 'configs/auto_optimized_config.yaml') -> str:
    """
    自动优化配置
    
    Args:
        base_config_path: 基础配置文件路径
        output_path: 输出配置文件路径
        
    Returns:
        优化后的配置文件路径
    """
    print("Starting automatic config optimization...")
    
    # 检测GPU
    monitor = GPUMonitor()
    if not torch.cuda.is_available():
        print("No CUDA device available, using CPU config")
        return base_config_path
    
    gpu_memory_gb = monitor.stats['memory_total']
    print(f"Detected GPU: {monitor.device_name}")
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # 创建优化配置
    optimizer = ConfigOptimizer(base_config_path)
    optimized_config = optimizer.create_optimized_config(gpu_memory_gb)
    
    # 保存配置
    optimizer.save_optimized_config(optimized_config, output_path)
    
    print("Config optimization completed!")
    print(f"Recommended settings:")
    print(f"  - Batch size: {optimized_config['batch_size']}")
    print(f"  - Base channels: {optimized_config['base_channels']}")
    print(f"  - Num workers: {optimized_config['num_workers']}")
    print(f"  - Mixed precision: {optimized_config['use_mixed_precision']}")
    
    return output_path

if __name__ == "__main__":
    # 示例用法
    optimized_config_path = auto_optimize_config()
    print(f"Use the optimized config: {optimized_config_path}") 