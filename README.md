# Lightweight Super-Resolution Project

A lightweight super-resolution project based on a roadmap, implementing a complete training process from foundational architecture to advanced optimizations.

## 🎯 Project Features

- **Lightweight Design**: Utilizes lightweight components such as PixelShuffle, BSConv, and ConvNeXt.
- **Interpolation-Free**: Abandons bicubic interpolation, directly processes LR inputs.
- **Multi-Stage Training**: Incrementally adds modules from basic to advanced for optimization.
- **Efficient Attention**: Integrates Omni-SR, spatial, and channel attention mechanisms.
- **Fast Convergence**: Uses LoRA/ConvLoRA and Adan optimizer for accelerated training.
- **Optional Enhancements**: Supports flexible addition of Transformer and GAN modules.

## 📋 Roadmap Implementation

### Step 1: Basic Architecture (Lightweight + Interpolation-Free)

- ✅ PixelShuffle upsampling structure
- ✅ BSConv (Depthwise Separable Convolution)
- ✅ ConvNeXt-style residual structure
- ✅ Abandon bicubic interpolation

### Step 2: Efficiency Enhancement Module Integration

- ✅ Omni-SR streamlined attention mechanism
- ✅ Spatial + channel attention
- ✅ Region-adaptive attention

### Step 3: Accelerated Training + Lightweight Fine-Tuning

- ✅ LoRA/ConvLoRA low-rank adaptation
- ✅ Distillation supervision
- ✅ Multi-stage warm-start strategy

### Step 4: Optional Innovative Modules

- ✅ Progressive focus Transformer (PFT)
- ✅ GAN-splice lightweight adversarial loss
- ✅ Region-adaptive attention

### Step 5: Training Strategy and Evaluation

- ✅ DIV2K/LSDIR dataset support
- ✅ PSNR/SSIM evaluation metrics
- ✅ AdamW/Adan optimizer
- ✅ Multi-stage training monitoring

## 🚀 Quick Start

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Create Configuration Files

```bash
python src/main.py --mode create_configs
```

This will create three preset configurations:

- `lightweight_config.yaml`: Lightweight configuration (fast training)
- `balanced_config.yaml`: Balanced configuration (recommended)
- `high_quality_config.yaml`: High-quality configuration (best performance)

### Data Preparation

Organize the dataset structure as follows:

```
data/
├── train/
│   ├── HR/          # High-resolution training images
│   └── LR/          # Low-resolution training images
├── val/
│   ├── HR/          # High-resolution validation images
│   └── LR/          # Low-resolution validation images
└── test/
    ├── HR/          # High-resolution test images
    └── LR/          # Low-resolution test images
```

### Start Training

```bash
# Train with balanced configuration
python src/main.py --mode train \
    --config configs/balanced_config.yaml \
    --data_dir data \
    --experiment_name my_experiment

# Quick training with lightweight configuration
python src/main.py --mode train \
    --config configs/lightweight_config.yaml \
    --data_dir data \
    --experiment_name quick_test
```

### Model Evaluation

```bash
# Evaluate trained model
python src/main.py --mode evaluate \
    --model_path models/best_stage_3.pth \
    --test_data_dir data/test
```

### Performance Benchmarking

```bash
# Test model inference speed and memory usage
python src/main.py --mode benchmark \
    --model_path models/best_stage_3.pth
```

### Inference

```bash
python inference.py --model models/stage_1_final.pth --input data/val/LR/00000.png --output result_stage1_00000.png
```

## 📁 Project Structure

```
├── src/
│   └── main.py              # Main entry file
├── modules/
│   ├── backbone.py          # Backbone network (PixelShuffle, BSConv, ConvNeXt)
│   ├── attention.py         # Attention mechanisms (Omni-SR, spatial + channel)
│   ├── lora.py              # LoRA/ConvLoRA fine-tuning
│   ├── transformer.py       # Transformer module (PFT)
│   └── gan.py               # GAN module (GAN-splice)
├── training/
│   ├── train.py             # Training script (multi-stage training)
│   └── evaluate.py          # Evaluation script (PSNR/SSIM)
├── utils/
│   └── config.py            # Configuration management (multi-stage warm-start)
├── data/                    # Dataset directory
├── models/                  # Saved models directory
├── experiments/             # Experiment results directory
├── configs/                 # Configuration files directory
├── tests/                   # Unit tests
└── requirements.txt         # Dependency list
```

## 🔧 Configuration Details

### Key Parameters

```yaml
# Model parameters
base_channels: 64
num_blocks: 8
scale_factor: 4

# Training parameters
batch_size: 16
lr: 1e-4
optimizer: adamw
epochs_per_stage: 100
num_stages: 4

# Loss weights
mse_weight: 1.0
perceptual_weight: 0.1
adversarial_weight: 0.001
```

### Multi-Stage Training

The project supports 4 training stages:

1. **Stage 0**: Basic backbone (PixelShuffle + BSConv + ConvNeXt)
2. **Stage 1**: Adding attention mechanisms (Omni-SR)
3. **Stage 2**: Adding Transformer module (PFT)
4. **Stage 3**: Adding GAN component (GAN-splice)

Each stage builds upon the results of the previous stage with warm-start training.

## 📊 Performance Metrics

### Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Performance Benchmarks

- **Inference Speed**: FPS (Frames Per Second)
- **Memory Usage**: GPU/CPU memory consumption
- **Model Size**: Parameter count and file size

## 🛠️ Custom Development

### Adding New Attention Mechanisms

```python
# Add in modules/attention.py
class CustomAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, x):
        return x
```

### Adding New Loss Functions

```python
# Add in training/train.py
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return loss
```

### Modifying Training Strategy

```python
# Modify in utils/config.py
class TrainingConfig:
    custom_param: float = 1.0
```

## 📈 Experiment Results

### Training Curves

- PSNR/SSIM changes per stage
- Loss function convergence
- Learning rate scheduling effects

### Model Comparisons

- Performance comparison across different configurations
- Comparison with SOTA methods
- Speed-quality trade-off analysis