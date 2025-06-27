import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Any, Tuple
import os
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import our modules
from modules.backbone import LightweightBackbone
from modules.attention import EfficientAttentionBlock
from modules.transformer import LightweightTransformer

class SuperResolutionDataset:
    """
    Dataset class for super-resolution training and evaluation.
    Supports DIV2K, LSDIR, and other common datasets.
    """
    def __init__(self, hr_dir: str, lr_dir: Optional[str] = None, 
                 scale_factor: int = 4, patch_size: int = 128, 
                 is_train: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train
        
        # Get image files
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if lr_dir:
            self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.lr_files = None
            
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        if self.lr_dir:
            # Load LR image if provided
            lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
            lr_img = cv2.imread(lr_path)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        else:
            # Generate LR image by downsampling
            h, w = hr_img.shape[:2]
            lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
            lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to tensor and normalize
        hr_tensor = torch.from_numpy(hr_img).float().permute(2, 0, 1) / 255.0
        lr_tensor = torch.from_numpy(lr_img).float().permute(2, 0, 1) / 255.0
        
        # Data augmentation for training
        if self.is_train:
            hr_tensor, lr_tensor = self._augment(hr_tensor, lr_tensor)
        
        return lr_tensor, hr_tensor
    
    def _augment(self, hr_tensor, lr_tensor):
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, [2])
            lr_tensor = torch.flip(lr_tensor, [2])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            hr_tensor = torch.flip(hr_tensor, [1])
            lr_tensor = torch.flip(lr_tensor, [1])
        
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            hr_tensor = torch.rot90(hr_tensor, k, [1, 2])
            lr_tensor = torch.rot90(lr_tensor, k, [1, 2])
        
        return hr_tensor, lr_tensor

class Evaluator:
    """
    Evaluator for super-resolution models.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def evaluate_dataset(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        """
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(test_loader, desc="Evaluating"):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Generate super-resolution images
                sr_imgs = self.model(lr_imgs)
                
                # Calculate metrics
                batch_psnr = self._calculate_psnr(sr_imgs, hr_imgs)
                batch_ssim = self._calculate_ssim(sr_imgs, hr_imgs)
                batch_lpips = self._calculate_lpips(sr_imgs, hr_imgs)
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_lpips += batch_lpips
                num_samples += lr_imgs.size(0)
        
        return {
            'PSNR': total_psnr / num_samples,
            'SSIM': total_ssim / num_samples,
            'LPIPS': total_lpips / num_samples
        }
    
    def _calculate_psnr(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """Calculate PSNR."""
        sr_imgs = torch.clamp(sr_imgs, 0, 1)
        hr_imgs = torch.clamp(hr_imgs, 0, 1)
        
        mse = F.mse_loss(sr_imgs, hr_imgs)
        if mse == 0:
            return float('inf')
        
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def _calculate_ssim(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """Calculate SSIM."""
        sr_imgs = torch.clamp(sr_imgs, 0, 1)
        hr_imgs = torch.clamp(hr_imgs, 0, 1)
        
        # Convert to numpy for skimage
        sr_np = sr_imgs.cpu().numpy()
        hr_np = hr_imgs.cpu().numpy()
        
        total_ssim = 0.0
        for i in range(sr_np.shape[0]):
            ssim_val = ssim(sr_np[i].transpose(1, 2, 0), 
                           hr_np[i].transpose(1, 2, 0), 
                           multichannel=True, data_range=1.0)
            total_ssim += ssim_val
        
        return total_ssim
    
    def _calculate_lpips(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor) -> float:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity)."""
        # Simplified LPIPS calculation
        # For production, use proper LPIPS implementation
        sr_imgs = torch.clamp(sr_imgs, 0, 1)
        hr_imgs = torch.clamp(hr_imgs, 0, 1)
        
        # Convert to [-1, 1] range
        sr_imgs = sr_imgs * 2 - 1
        hr_imgs = hr_imgs * 2 - 1
        
        # Simple perceptual loss using VGG features
        # This is a placeholder - implement proper LPIPS
        return F.mse_loss(sr_imgs, hr_imgs).item()

def evaluate_model(model_path: str, test_data_dir: str, 
                  scale_factor: int = 4, batch_size: int = 1) -> Dict[str, float]:
    """
    Evaluate a trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Build model
    model = LightweightBackbone(
        in_channels=3,
        base_channels=config['base_channels'],
        num_blocks=config['num_blocks'],
        scale_factor=config['scale_factor']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create test dataset
    test_dataset = SuperResolutionDataset(
        hr_dir=os.path.join(test_data_dir, 'HR'),
        lr_dir=os.path.join(test_data_dir, 'LR'),
        scale_factor=scale_factor,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate_dataset(test_loader)
    
    return results

def benchmark_model(model_path: str, input_size: Tuple[int, int] = (256, 256),
                   num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark model inference speed and memory usage.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = LightweightBackbone(
        in_channels=3,
        base_channels=config['base_channels'],
        num_blocks=config['num_blocks'],
        scale_factor=config['scale_factor']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_inference_time': avg_time,
        'fps': fps,
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved
    }

def compare_models(model_paths: list, test_data_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test dataset.
    """
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).split('.')[0]
        print(f"Evaluating {model_name}...")
        
        try:
            model_results = evaluate_model(model_path, test_data_dir)
            results[model_name] = model_results
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

# Import time for benchmarking
import time 