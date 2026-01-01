import timm
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
from torch import nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from improvements.mixup import mixup_collate_fn

@dataclass
class ModelConfig:
    model_name: str = 'tinynet_e.in1k'
    pretrained: bool = False
    num_classes: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


class ImageWoofDataset(Dataset):
    def __init__(self, data_dir: Path, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get class folders (excluding .DS_Store)
        self.class_folders = sorted([d for d in data_dir.iterdir() 
                                     if d.is_dir() and d.name != '.DS_Store'])
        
        # Create class name to index mapping
        self.classes = {folder.name: idx for idx, folder in enumerate(self.class_folders)}
        
        # Collect all image paths with their labels
        self.samples = []
        for class_folder in self.class_folders:
            class_idx = self.classes[class_folder.name]
            # Find all JPEG/jpg images in this class folder
            image_files = list(class_folder.glob('*.JPEG')) + list(class_folder.glob('*.jpg'))
            for img_path in image_files:
                self.samples.append((img_path, class_idx))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Read image with cv2 (returns numpy array - perfect for Albumentations!)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label


class ClassificationModel(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        
        # Model setup
        self.model = timm.create_model(
            config.model_name, 
            pretrained=config.pretrained, 
            num_classes=config.num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
            # Unpack the dict from our custom collate_fn
            x = batch["x"]
            y_a, y_b = batch["y_a"], batch["y_b"]
            lam = batch["lam"]
            
            y_hat = self(x)
            
            # Mixup Loss: weighted sum of two cross-entropy calculations
            loss = lam * self.criterion(y_hat, y_a) + (1 - lam) * self.criterion(y_hat, y_b)
            
            # For accuracy, we usually just compare against the dominant label (y_a)
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y_a).float().mean()
            
            self.log('train_loss', loss, on_epoch=True, prog_bar=True)
            self.log('train_acc', acc, on_epoch=True, prog_bar=True)
            return loss
            
    
    def validation_step(self, batch, batch_idx):
            # Validation batch is still a standard (x, y) tuple
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
            
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_epoch=True, prog_bar=True)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
if __name__ == '__main__':
    # Initialize model with config
    config = ModelConfig(
        model_name='tinynet_e.in1k',
        num_classes=10,
        learning_rate=1e-3,
        weight_decay=0.01,
        pretrained=True,
    )

    model = ClassificationModel(config)

    # Define Albumentations transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()  # Converts to PyTorch tensor
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    # Create datasets with Albumentations transforms
    train_path = Path('/Users/jeremyong/Desktop/research_agent/dataset/imagewoof-160/train')
    val_path = Path('/Users/jeremyong/Desktop/research_agent/dataset/imagewoof-160/val')

    train_dataset = ImageWoofDataset(train_path, transform=train_transform)
    val_dataset = ImageWoofDataset(val_path, transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=mixup_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Train
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        logger=WandbLogger(project="imagewoof-160", name="tinynet_mixup"),
        callbacks=[
            ModelCheckpoint(
                monitor='val_acc', 
                mode='max', 
                save_top_k=3,
                filename='tinynet-{epoch:02d}-{val_acc:.4f}'
            )
        ]
    )

    trainer.fit(model, train_loader, val_loader)