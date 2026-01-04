"""
MMPretrain version of vit.py

This script uses MMPretrain (formerly MMClassification) to train TinyNet with Mixup
on the ImageWoof dataset.

Usage:
    python classification/classification_models/vit_mmpretrain.py

Features:
- TinyNet-E model from TIMM
- Mixup augmentation (alpha=0.2)
- AdamW optimizer with cosine annealing
- WandB logging
- Automatic checkpointing
"""

import sys
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
mmpretrain_path = project_root / 'mmpretrain'
sys.path.insert(0, str(mmpretrain_path))
sys.path.insert(0, str(project_root))  # Add classification folder to path

from mmengine.config import Config
from mmengine.runner import Runner
import mmpretrain

# Import custom transforms to register them
try:
    from improvements.gabor_transform import GaborFilter
    print("✓ GaborFilter registered")
except ImportError:
    pass  # Not using custom transforms

def print_dataset_info(cfg):
    """Print dataset information similar to vit.py"""
    train_root = cfg.train_dataloader.dataset.data_root
    val_root = cfg.val_dataloader.dataset.data_root
    
    train_path = Path(train_root)
    val_path = Path(val_root)
    
    # Count training samples
    train_samples = 0
    train_classes = {}
    if train_path.exists():
        class_folders = sorted([d for d in train_path.iterdir() if d.is_dir()])
        for idx, class_folder in enumerate(class_folders):
            train_classes[class_folder.name] = idx
            train_samples += len(list(class_folder.glob('*.JPEG')) + list(class_folder.glob('*.jpg')))
    
    # Count validation samples
    val_samples = 0
    if val_path.exists():
        class_folders = sorted([d for d in val_path.iterdir() if d.is_dir()])
        for class_folder in class_folders:
            val_samples += len(list(class_folder.glob('*.JPEG')) + list(class_folder.glob('*.jpg')))
    
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Number of classes: {len(train_classes)}")
    print(f"Classes: {train_classes}")
    print("=" * 60)
    print()


def print_model_info(cfg):
    """Print model configuration"""
    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    
    # Handle config dict access safely
    try:
        if hasattr(cfg.model, 'backbone') and 'model_name' in cfg.model.backbone:
            print(f"Model: {cfg.model.backbone.model_name}")
            print(f"Pretrained: {cfg.model.backbone.pretrained}")
        else:
            print(f"Model type: {cfg.model.get('type', 'Unknown')}")
    except (AttributeError, KeyError):
        print(f"Model type: {cfg.model.get('type', 'Unknown')}")
    
    try:
        if hasattr(cfg.model, 'head') and 'num_classes' in cfg.model.head:
            print(f"Number of classes: {cfg.model.head.num_classes}")
    except (AttributeError, KeyError):
        pass
    
    print(f"Learning rate: {cfg.optim_wrapper.optimizer.lr}")
    print(f"Weight decay: {cfg.optim_wrapper.optimizer.weight_decay}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"Max epochs: {cfg.train_cfg.max_epochs}")
    
    if hasattr(cfg.model, 'train_cfg') and cfg.model.train_cfg and 'augments' in cfg.model.train_cfg:
        print(f"Augmentations: {[aug['type'] for aug in cfg.model.train_cfg.augments]}")
    
    print("=" * 60)
    print()


def main():
    # Load config (equivalent to vit.py's ModelConfig and hyperparameters)
    config_file = 'configs/tinynet_mixup_imagewoof.py'
    config_path = Path(__file__).parent / config_file
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print(f"Please run this script from the project root or ensure the config exists.")
        return
    
    cfg = Config.fromfile(str(config_path))
    
    # Set work directory (equivalent to PyTorch Lightning's default_root_dir)
    cfg.work_dir = './work_dirs/tinynet_mixup_imagewoof'
    
    # Print information (equivalent to vit.py's print statements)
    print_dataset_info(cfg)
    print_model_info(cfg)
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Work directory: {cfg.work_dir}")
    print(f"Logs and checkpoints will be saved to: {cfg.work_dir}")
    print("=" * 60)
    print()
    
    # Build the runner (equivalent to pl.Trainer)
    runner = Runner.from_cfg(cfg)
    
    # Start training (equivalent to trainer.fit())
    runner.train()
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {cfg.work_dir}")
    print(f"Best checkpoint: {cfg.work_dir}/best_accuracy_top1_*.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()

