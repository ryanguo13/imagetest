#!/usr/bin/env python3
"""
测试ConfigManager是否能正确读取早停配置
"""

from utils.config import ConfigManager

def test_config_manager():
    print("🧪 测试ConfigManager读取早停配置...")
    
    try:
        # 测试早停配置文件
        cm = ConfigManager('configs/early_stopping_config.yaml')
    except Exception as e:
        print(f"❌ 创建ConfigManager失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print(f"Early stopping enabled: {cm.config.early_stopping_enabled}")
        print(f"Patience: {cm.config.early_stopping_patience}")
        print(f"Min delta: {cm.config.early_stopping_min_delta}")
        print(f"Monitor: {cm.config.early_stopping_monitor}")
        print(f"Restore weights: {cm.config.early_stopping_restore_weights}")
        
        # 验证配置正确性
        assert cm.config.early_stopping_enabled == True, "早停应该启用"
        assert cm.config.early_stopping_patience == 10, "Patience应该是10"
        assert cm.config.early_stopping_monitor == 'psnr', "Monitor应该是psnr"
        
        print("✅ ConfigManager读取早停配置成功！")
        
        # 测试转换为字典
        from dataclasses import asdict
        config_dict = asdict(cm.config)
        
        print(f"\n字典中的早停配置:")
        for key, value in config_dict.items():
            if 'early_stopping' in key:
                print(f"  {key}: {value}")
        
        print("✅ 配置字典转换成功！")
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_manager() 