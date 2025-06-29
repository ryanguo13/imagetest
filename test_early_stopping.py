#!/usr/bin/env python3
"""
早停机制测试脚本
"""

import torch
import sys
import os
sys.path.append('.')

from training.train import EarlyStopping

def test_early_stopping_basic():
    """测试基本早停功能"""
    print("🧪 测试早停机制...")
    
    # 测试PSNR监控（越大越好）
    early_stopping = EarlyStopping(patience=3, min_delta=0.1, mode='max')
    
    # 模拟训练过程
    scores = [25.0, 25.5, 26.0, 26.8, 26.7, 26.6, 26.5, 26.4, 26.3, 26.2]
    
    print("模拟PSNR变化:", scores)
    
    for epoch, score in enumerate(scores):
        should_stop = early_stopping(score)
        print(f"Epoch {epoch}: PSNR={score:.1f}, Counter={early_stopping.counter}, Should stop: {should_stop}")
        
        if should_stop:
            print(f"✅ 早停在第{epoch}轮触发，最佳PSNR: {early_stopping.get_best_score():.1f}")
            break
    
    print()

def test_early_stopping_loss():
    """测试Loss监控（越小越好）"""
    print("🧪 测试Loss监控...")
    
    early_stopping = EarlyStopping(patience=4, min_delta=0.001, mode='min')
    
    # 模拟训练损失变化
    losses = [0.1, 0.08, 0.06, 0.055, 0.054, 0.0535, 0.0534, 0.0533, 0.0532]
    
    print("模拟Loss变化:", losses)
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Counter={early_stopping.counter}, Should stop: {should_stop}")
        
        if should_stop:
            print(f"✅ 早停在第{epoch}轮触发，最佳Loss: {early_stopping.get_best_score():.4f}")
            break
    
    print()

def test_config_loading():
    """测试配置加载"""
    print("🧪 测试配置文件加载...")
    
    import yaml
    
    try:
        # 测试加载配置文件
        with open('configs/early_stopping_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查早停参数
        early_stopping_params = {
            'early_stopping_enabled': config.get('early_stopping_enabled'),
            'early_stopping_patience': config.get('early_stopping_patience'),
            'early_stopping_min_delta': config.get('early_stopping_min_delta'),
            'early_stopping_monitor': config.get('early_stopping_monitor'),
            'early_stopping_restore_weights': config.get('early_stopping_restore_weights')
        }
        
        print("✅ 配置文件加载成功:")
        for key, value in early_stopping_params.items():
            print(f"  {key}: {value}")
        
        # 验证配置
        assert config['early_stopping_enabled'] == True, "早停应该启用"
        assert config['early_stopping_patience'] == 10, f"Patience应该是10，实际是{config['early_stopping_patience']}"
        assert config['early_stopping_monitor'] == 'psnr', f"Monitor应该是psnr，实际是{config['early_stopping_monitor']}"
        
        print("✅ 所有配置验证通过")
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
    
    print()

def test_weight_restoration():
    """测试权重恢复功能"""
    print("🧪 测试权重恢复...")
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 1)
    original_weight = model.weight.data.clone()
    
    early_stopping = EarlyStopping(patience=2, mode='max', restore_best_weights=True)
    
    # 第一次：最佳权重
    best_score = 0.9
    early_stopping(best_score, model)
    
    # 修改权重（模拟继续训练）
    model.weight.data.fill_(999.0)
    
    # 第二次和第三次：性能下降
    early_stopping(0.8, model)
    should_stop = early_stopping(0.7, model)
    
    if should_stop:
        # 恢复最佳权重
        early_stopping.restore_weights(model)
        
        # 检查权重是否恢复
        if not torch.equal(model.weight.data, torch.full_like(model.weight.data, 999.0)):
            print("✅ 权重成功恢复到最佳状态")
        else:
            print("❌ 权重恢复失败")
    
    print()

def main():
    print("🚀 早停机制测试套件")
    print("=" * 50)
    
    test_early_stopping_basic()
    test_early_stopping_loss()
    test_config_loading()
    test_weight_restoration()
    
    print("🎉 所有测试完成!")

if __name__ == "__main__":
    main() 