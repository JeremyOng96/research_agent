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
    import argparse
    import json
    from torch.utils.data import Subset
    
    parser = argparse.ArgumentParser(description='Train TinyNet on ImageWoof')
    parser.add_argument(
        '--config',
        default='configs/tinynet_mixup_imagewoof.py',
        help='Config file path (relative to this script)'
    )
    parser.add_argument(
        '--use-cleaned-dataset',
        action='store_true',
        help='Use cleaned dataset (filter out 533 issue indices from Cleanlab)'
    )
    args = parser.parse_args()
    
    # Load config (equivalent to vit.py's ModelConfig and hyperparameters)
    config_path = Path(__file__).parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print(f"Please run this script from the project root or ensure the config exists.")
        return
    
    cfg = Config.fromfile(str(config_path))
    
    # Load clean indices if using cleaned dataset
    clean_indices = None
    if args.use_cleaned_dataset:
        json_path = Path(__file__).parent / 'configs' / 'clean_dataset.json'
        if not json_path.exists():
            print(f"Error: clean_dataset.json not found at {json_path}")
            print("Please run the cleaning agent first to generate this file.")
            return
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        issue_indices = set(metadata['issue_indices'])
        total_samples = metadata['total_samples']
        clean_indices = [i for i in range(total_samples) if i not in issue_indices]
        
        print("\n" + "="*60)
        print("USING CLEANED DATASET")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Issue samples (removed): {len(issue_indices)}")
        print(f"Clean samples (used): {len(clean_indices)}")
        print("="*60 + "\n")
        
        # Update work_dir if not already set to cleaned version
        if 'cleaned' not in str(cfg.get('work_dir', '')):
            base_work_dir = cfg.get('work_dir', './work_dirs/tinynet_mixup_imagewoof')
            cfg.work_dir = str(Path(base_work_dir).parent / f"{Path(base_work_dir).name}_cleaned")
    
    # Set work directory (equivalent to PyTorch Lightning's default_root_dir)
    if not hasattr(cfg, 'work_dir') or cfg.work_dir is None:
        cfg.work_dir = './work_dirs/tinynet_mixup_imagewoof'
    
    # Print information (equivalent to vit.py's print statements)
    print_dataset_info(cfg)
    print_model_info(cfg)
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Work directory: {cfg.work_dir}")
    print(f"Config: {args.config}")
    print(f"Logs and checkpoints will be saved to: {cfg.work_dir}")
    print("=" * 60)
    print()
    
    # Build the runner first (equivalent to pl.Trainer)
    runner = Runner.from_cfg(cfg)
    
    # Apply dataset filtering AFTER runner is built
    if clean_indices is not None:
        from torch.utils.data import DataLoader
        
        print("\nApplying dataset filtering...")
        original_dataset = runner.train_dataloader.dataset
        original_size = len(original_dataset)
        filtered_dataset = Subset(original_dataset, clean_indices)
        
        # Get dataloader config
        batch_size = runner.train_dataloader.batch_size
        num_workers = runner.train_dataloader.num_workers
        
        # Create a new dataloader with the filtered dataset
        # We need to rebuild the dataloader because PyTorch DataLoaders are immutable
        new_dataloader = DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            collate_fn=runner.train_dataloader.collate_fn,
        )
        
        # Replace the entire dataloader (not just the dataset)
        runner._train_dataloader = new_dataloader
        
        print(f"✓ Filtered training dataset: {original_size} → {len(filtered_dataset)} samples\n")
    
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

