"""
Example: Loading dataset from MMPretrain config

This shows how to load the dataset defined in your config file
for use with cleanlab or other data cleaning tools.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'classification' / 'mmpretrain'))

from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmpretrain.datasets import build_dataset
from mmpretrain.registry import DATASETS


def load_dataset_from_config(config_path: str, split: str = 'train'):
    """
    Load dataset from MMPretrain config file.
    
    Args:
        config_path: Path to config file
        split: 'train' or 'val'
    
    Returns:
        dataset: MMPretrain dataset
        dataloader: PyTorch DataLoader
    """
    # 1. Load config
    cfg = Config.fromfile(config_path)
    
    # 2. Get dataset config based on split
    if split == 'train':
        dataset_cfg = cfg.train_dataloader.dataset
        dataloader_cfg = cfg.train_dataloader
    elif split == 'val':
        dataset_cfg = cfg.val_dataloader.dataset
        dataloader_cfg = cfg.val_dataloader
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split}")
    
    # 3. Build dataset
    dataset = build_dataset(dataset_cfg)
    
    # 4. Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cfg.get('batch_size', 32),
        num_workers=dataloader_cfg.get('num_workers', 4),
        shuffle=(split == 'train'),
        drop_last=False,
        persistent_workers=False  # Set to False for compatibility
    )
    
    return dataset, dataloader


def get_dataset_info(dataset):
    """Print dataset information."""
    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Data root: {dataset.data_root}")
    
    # Get first sample to see structure
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    if 'img' in sample:
        print(f"Image shape: {sample['img'].shape}")
    if 'gt_label' in sample:
        print(f"Label: {sample['gt_label']}")


# Example usage
if __name__ == "__main__":
    # Path to your config
    config_path = 'classification/classification_models/configs/tinynet_mixup_imagewoof.py'
    
    # Load training dataset
    print("Loading training dataset...")
    train_dataset, train_loader = load_dataset_from_config(config_path, split='train')
    get_dataset_info(train_dataset)
    
    print("\n" + "="*60 + "\n")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset, val_loader = load_dataset_from_config(config_path, split='val')
    get_dataset_info(val_dataset)
    
    # Iterate through a few batches
    print("\n" + "="*60)
    print("Iterating through batches...")
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Just show first 2 batches
            break
        print(f"\nBatch {i}:")
        print(f"  Data keys: {batch.keys()}")
        if 'inputs' in batch:
            print(f"  Images shape: {batch['inputs'].shape}")
        if 'data_samples' in batch:
            print(f"  Labels: {[ds.gt_label.item() for ds in batch['data_samples'][:5]]}...")

