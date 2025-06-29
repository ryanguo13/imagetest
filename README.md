# Lightweight Super-Resolution Project

一个基于路线图的轻量级超分辨率项目，实现了从基础架构到高级优化的完整训练流程。

## 🎯 项目特点

- **轻量设计**: 使用PixelShuffle、BSConv、ConvNeXt等轻量级组件
- **无插值**: 抛弃bicubic插值，直接从LR输入开始处理
- **多阶段训练**: 渐进式添加模块，从基础到高级逐步优化
- **高效注意力**: 集成Omni-SR、空间+通道注意力机制
- **快速收敛**: 使用LoRA/ConvLoRA和Adan优化器加速训练
- **可选增强**: 支持Transformer和GAN模块的灵活添加

## 📋 路线图实现

### Step 1: 基础架构（轻量＋无插值）
- ✅ PixelShuffle上采样结构
- ✅ BSConv（深度可分离卷积）
- ✅ ConvNeXt风格残差结构
- ✅ 抛弃bicubic插值

### Step 2: 效率提升模块集成
- ✅ Omni-SR精简注意力机制
- ✅ 空间+通道注意力
- ✅ 区域自适应注意力

### Step 3: 加速训练＋轻量微调
- ✅ LoRA/ConvLoRA低秩适配
- ✅ 蒸馏监督
- ✅ 多阶段warm-start策略

### Step 4: 可选创新增强模块
- ✅ 渐进聚焦Transformer（PFT）
- ✅ GAN-splice轻量对抗loss
- ✅ 区域自适应注意力

### Step 5: 训练策略与评估
- ✅ DIV2K/LSDIR数据集支持
- ✅ PSNR/SSIM评估指标
- ✅ AdamW/Adan优化器
- ✅ 多阶段训练监控

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 创建配置文件

```bash
python src/main.py --mode create_configs
```

这将创建三个预设配置：
- `lightweight_config.yaml`: 轻量级配置（快速训练）
- `balanced_config.yaml`: 平衡配置（推荐）
- `high_quality_config.yaml`: 高质量配置（最佳效果）

### 数据准备

将数据集按以下结构组织：

```
data/
├── train/
│   ├── HR/          # 高分辨率训练图像
│   └── LR/          # 低分辨率训练图像
├── val/
│   ├── HR/          # 高分辨率验证图像
│   └── LR/          # 低分辨率验证图像
└── test/
    ├── HR/          # 高分辨率测试图像
    └── LR/          # 低分辨率测试图像
```

### 开始训练

```bash
# 使用平衡配置训练
python src/main.py --mode train \
    --config configs/balanced_config.yaml \
    --data_dir data \
    --experiment_name my_experiment

# 使用轻量级配置快速训练
python src/main.py --mode train \
    --config configs/lightweight_config.yaml \
    --data_dir data \
    --experiment_name quick_test
```

### 模型评估

```bash
# 评估训练好的模型
python src/main.py --mode evaluate \
    --model_path models/best_stage_3.pth \
    --test_data_dir data/test
```

### 性能基准测试

```bash
# 测试模型推理速度和内存使用
python src/main.py --mode benchmark \
    --model_path models/best_stage_3.pth
```

### Inference

```bash
python inference.py --model models/stage_1_final.pth --input data/val/LR/00000.png --output result_stage1_00000.png

```

## 📁 项目结构

```
├── src/
│   └── main.py              # 主入口文件
├── modules/
│   ├── backbone.py          # 主干网络（PixelShuffle, BSConv, ConvNeXt）
│   ├── attention.py         # 注意力机制（Omni-SR, 空间+通道）
│   ├── lora.py             # LoRA/ConvLoRA微调
│   ├── transformer.py      # Transformer模块（PFT）
│   └── gan.py              # GAN模块（GAN-splice）
├── training/
│   ├── train.py            # 训练脚本（多阶段训练）
│   └── evaluate.py         # 评估脚本（PSNR/SSIM）
├── utils/
│   └── config.py           # 配置管理（多阶段warm-start）
├── data/                   # 数据集目录
├── models/                 # 模型保存目录
├── experiments/            # 实验结果目录
├── configs/                # 配置文件目录
├── tests/                  # 单元测试
└── requirements.txt        # 依赖包列表
```

## 🔧 配置说明

### 主要参数

```yaml
# 模型参数
base_channels: 64          # 基础通道数
num_blocks: 8              # 残差块数量
scale_factor: 4            # 超分辨率倍数

# 训练参数
batch_size: 16             # 批次大小
lr: 1e-4                   # 学习率
optimizer: adamw           # 优化器类型
epochs_per_stage: 100      # 每阶段训练轮数
num_stages: 4              # 训练阶段数

# 损失权重
mse_weight: 1.0            # MSE损失权重
perceptual_weight: 0.1     # 感知损失权重
adversarial_weight: 0.001  # 对抗损失权重
```

### 多阶段训练

项目支持4个训练阶段：

1. **Stage 0**: 基础backbone（PixelShuffle + BSConv + ConvNeXt）
2. **Stage 1**: 添加注意力机制（Omni-SR）
3. **Stage 2**: 添加Transformer模块（PFT）
4. **Stage 3**: 添加GAN组件（GAN-splice）

每个阶段都会基于前一阶段的结果进行warm-start训练。

## 📊 性能指标

### 评估指标
- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数
- **LPIPS**: 学习感知图像块相似度

### 性能基准
- **推理速度**: FPS（帧每秒）
- **内存使用**: GPU/CPU内存占用
- **模型大小**: 参数量和文件大小

## 🛠️ 自定义开发

### 添加新的注意力机制

```python
# 在 modules/attention.py 中添加
class CustomAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 实现你的注意力机制
        
    def forward(self, x):
        # 实现前向传播
        return x
```

### 添加新的损失函数

```python
# 在 training/train.py 中添加
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # 实现你的损失函数
        return loss
```

### 修改训练策略

```python
# 在 utils/config.py 中修改
class TrainingConfig:
    # 添加新的配置参数
    custom_param: float = 1.0
```

## 📈 实验结果

### 训练曲线
- 每个阶段的PSNR/SSIM变化
- 损失函数收敛情况
- 学习率调度效果

### 模型对比
- 不同配置的性能对比
- 与SOTA方法的比较
- 速度-质量权衡分析

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [LSDIR Dataset](https://github.com/csjliang/LSIR)
- [PyTorch](https://pytorch.org/)
- [scikit-image](https://scikit-image.org/)

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 这是一个研究项目，建议在实验环境中使用。生产环境使用前请充分测试。 