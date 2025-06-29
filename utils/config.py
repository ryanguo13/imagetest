import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    # Model parameters
    base_channels: int = 64
    num_blocks: int = 8
    scale_factor: int = 4
    
    # Training parameters
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = 'adamw'  # 'adamw' or 'adan'
    epochs_per_stage: int = 100
    num_stages: int = 4
    
    # Loss weights
    mse_weight: float = 1.0
    perceptual_weight: float = 0.1
    adversarial_weight: float = 0.001
    
    # Data parameters
    patch_size: int = 128
    num_workers: int = 4
    
    # Training strategy
    use_perceptual: bool = False
    use_gan: bool = False
    grad_clip: float = 0.0
    val_freq: int = 1
    
    # Multi-stage parameters
    stage_lr_decay: float = 0.5
    stage_attention: bool = True
    stage_transformer: bool = True
    stage_gan: bool = True
    
    # Early stopping parameters
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = 'psnr'
    early_stopping_restore_weights: bool = True

class ConfigManager:
    """
    Configuration manager for multi-stage training.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = TrainingConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
        
        # Update config
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, config_path: str):
        """Save configuration to file."""
        config_dict = asdict(self.config)
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """Get configuration for specific training stage."""
        config_dict = asdict(self.config)
        
        # Stage-specific modifications
        if stage == 0:
            # Basic backbone
            config_dict['use_perceptual'] = False
            config_dict['use_gan'] = False
            config_dict['stage_attention'] = False
            config_dict['stage_transformer'] = False
            config_dict['stage_gan'] = False
            
        elif stage == 1:
            # Add attention
            config_dict['use_perceptual'] = False
            config_dict['use_gan'] = False
            config_dict['stage_attention'] = True
            config_dict['stage_transformer'] = False
            config_dict['stage_gan'] = False
            config_dict['lr'] *= self.config.stage_lr_decay
            
        elif stage == 2:
            # Add transformer
            config_dict['use_perceptual'] = True
            config_dict['use_gan'] = False
            config_dict['stage_attention'] = True
            config_dict['stage_transformer'] = True
            config_dict['stage_gan'] = False
            config_dict['lr'] *= (self.config.stage_lr_decay ** 2)
            
        elif stage == 3:
            # Add GAN
            config_dict['use_perceptual'] = True
            config_dict['use_gan'] = True
            config_dict['stage_attention'] = True
            config_dict['stage_transformer'] = True
            config_dict['stage_gan'] = True
            config_dict['lr'] *= (self.config.stage_lr_decay ** 3)
        
        return config_dict
    
    def create_default_configs(self):
        """Create default configuration files for different scenarios."""
        configs = {
            'lightweight': TrainingConfig(
                base_channels=32,
                num_blocks=4,
                scale_factor=4,
                batch_size=32,
                lr=2e-4,
                epochs_per_stage=50,
                use_perceptual=False,
                use_gan=False
            ),
            'balanced': TrainingConfig(
                base_channels=64,
                num_blocks=8,
                scale_factor=4,
                batch_size=16,
                lr=1e-4,
                epochs_per_stage=100,
                use_perceptual=True,
                use_gan=False
            ),
            'high_quality': TrainingConfig(
                base_channels=128,
                num_blocks=12,
                scale_factor=4,
                batch_size=8,
                lr=5e-5,
                epochs_per_stage=150,
                use_perceptual=True,
                use_gan=True
            )
        }
        
        for name, config in configs.items():
            config_path = f"configs/{name}_config.yaml"
            os.makedirs("configs", exist_ok=True)
            
            config_dict = asdict(config)
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            print(f"Created {config_path}")

class WarmStartManager:
    """
    Manages warm-start strategy for multi-stage training.
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.current_stage = 0
        self.stage_checkpoints = {}
    
    def get_warm_start_config(self, stage: int) -> Dict[str, Any]:
        """Get warm-start configuration for current stage."""
        stage_config = self.config_manager.get_stage_config(stage)
        
        # Add warm-start specific parameters
        if stage > 0:
            stage_config['warm_start'] = True
            stage_config['warm_start_path'] = f"models/stage_{stage-1}_final.pth"
            stage_config['freeze_backbone'] = False
            stage_config['lr_multiplier'] = 0.1  # Lower LR for fine-tuning
        else:
            stage_config['warm_start'] = False
            stage_config['freeze_backbone'] = False
            stage_config['lr_multiplier'] = 1.0
        
        return stage_config
    
    def should_advance_stage(self, current_metrics: Dict[str, float], 
                           stage: int) -> bool:
        """Determine if should advance to next stage."""
        if stage >= self.config_manager.config.num_stages - 1:
            return False
        
        # Check if current stage has converged
        if 'PSNR' in current_metrics:
            psnr = current_metrics['PSNR']
            
            # Stage-specific advancement criteria
            if stage == 0 and psnr > 30.0:  # Basic backbone
                return True
            elif stage == 1 and psnr > 32.0:  # With attention
                return True
            elif stage == 2 and psnr > 33.0:  # With transformer
                return True
            elif stage == 3 and psnr > 34.0:  # With GAN
                return True
        
        return False
    
    def get_learning_rate_schedule(self, stage: int) -> Dict[str, float]:
        """Get learning rate schedule for current stage."""
        base_lr = self.config_manager.config.lr
        
        if stage == 0:
            # Initial training with high LR
            return {
                'initial_lr': base_lr,
                'min_lr': base_lr * 0.1,
                'decay_factor': 0.5,
                'decay_patience': 20
            }
        else:
            # Fine-tuning with lower LR
            return {
                'initial_lr': base_lr * (self.config_manager.config.stage_lr_decay ** stage),
                'min_lr': base_lr * 0.01,
                'decay_factor': 0.7,
                'decay_patience': 10
            }

class ExperimentTracker:
    """
    Track training experiments and results.
    """
    def __init__(self, experiment_name: str, config_manager: ConfigManager):
        self.experiment_name = experiment_name
        self.config_manager = config_manager
        self.results = {}
        
        # Create experiment directory
        self.exp_dir = f"experiments/{experiment_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Save initial config
        self.config_manager.save_config(f"{self.exp_dir}/config.yaml")
    
    def log_stage_results(self, stage: int, metrics: Dict[str, float]):
        """Log results for a training stage."""
        self.results[f"stage_{stage}"] = metrics
        
        # Save results
        results_path = f"{self.exp_dir}/stage_{stage}_results.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def log_final_results(self, final_metrics: Dict[str, float]):
        """Log final experiment results."""
        self.results['final'] = final_metrics
        
        # Save final results
        final_path = f"{self.exp_dir}/final_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    def get_best_stage(self) -> int:
        """Get the stage with best PSNR."""
        best_psnr = 0
        best_stage = 0
        
        for stage_key, metrics in self.results.items():
            if 'PSNR' in metrics and metrics['PSNR'] > best_psnr:
                best_psnr = metrics['PSNR']
                best_stage = int(stage_key.split('_')[1])
        
        return best_stage
    
    def generate_report(self) -> str:
        """Generate experiment report."""
        report = f"Experiment: {self.experiment_name}\n"
        report += "=" * 50 + "\n\n"
        
        for stage_key, metrics in self.results.items():
            report += f"{stage_key.upper()}:\n"
            for metric, value in metrics.items():
                report += f"  {metric}: {value:.4f}\n"
            report += "\n"
        
        # Save report
        report_path = f"{self.exp_dir}/report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report

# Default configuration
DEFAULT_CONFIG = {
    'base_channels': 64,
    'num_blocks': 8,
    'scale_factor': 4,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 0.01,
    'optimizer': 'adamw',
    'epochs_per_stage': 100,
    'num_stages': 4,
    'mse_weight': 1.0,
    'perceptual_weight': 0.1,
    'adversarial_weight': 0.001,
    'patch_size': 128,
    'num_workers': 4,
    'use_perceptual': False,
    'use_gan': False,
    'grad_clip': 0.0,
    'val_freq': 1,
    'stage_lr_decay': 0.5,
    'stage_attention': True,
    'stage_transformer': True,
    'stage_gan': True,
    # Early stopping parameters
    'early_stopping_enabled': False,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'early_stopping_monitor': 'psnr',
    'early_stopping_restore_weights': True
} 