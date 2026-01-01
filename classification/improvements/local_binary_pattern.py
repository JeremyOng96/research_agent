import torch
import torch.nn.functional as F
from improvements.improvement_method import ImprovementMethod


class LocalBinaryPattern(ImprovementMethod):
    """Local Binary Pattern augmentation method with built-in transformation."""
    
    @staticmethod
    def _apply_lbp_transform(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
        """
        Apply LBP transformation to a batch of images.
        
        Args:
            x: Batch of images (B, C, H, W)
            radius: LBP radius parameter
        
        Returns:
            LBP transformed images (B, C, H, W)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")

        # Define the 8 neighbors for a standard 3x3 LBP (Radius 1)
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),   (1, 1),  (1, 0),
            (1, -1),  (0, -1)
        ]

        b, c, h, w = x.shape
        padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
        lbp_output = torch.zeros_like(x, dtype=torch.float32)

        # Iterate through the 8 neighbors
        for i, (dy, dx) in enumerate(offsets):
            weight = 2**i
            y_start = radius + dy
            x_start = radius + dx
            neighbor_window = padded[:, :, y_start:y_start + h, x_start:x_start + w]
            mask = (neighbor_window >= x).float()
            lbp_output += mask * weight
            
        return lbp_output
    
    @staticmethod
    def collate_train_step_fn(batch, **kwargs):
        """
        Apply LBP transformation to training batch.
        
        Args:
            batch: List of (image, label) tuples
            **kwargs: 
                - apply_lbp (bool): Whether to apply LBP. Default: True
                - radius (int): LBP radius. Default: 1
        """
        apply_lbp = kwargs.get('apply_lbp', True)
        radius = kwargs.get('radius', 1)
        
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        # Apply LBP transformation if requested
        if apply_lbp:
            x = LocalBinaryPattern._apply_lbp_transform(x, radius=radius)
            x = x / 255.0  # Normalize to [0, 1]
        
        return {
            "x": x,
            "y": y
        }

    @staticmethod
    def collate_val_step_fn(batch, **kwargs):
        """
        Apply LBP transformation to validation batch.
        
        Args:
            batch: List of (image, label) tuples
            **kwargs:
                - apply_lbp (bool): Whether to apply LBP. Default: True
                - radius (int): LBP radius. Default: 1
        """
        apply_lbp = kwargs.get('apply_lbp', True)
        radius = kwargs.get('radius', 1)
        
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        # Apply LBP transformation if requested
        if apply_lbp:
            x = LocalBinaryPattern._apply_lbp_transform(x, radius=radius)
            x = x / 255.0  # Normalize to [0, 1]
        
        return {
            "x": x,
            "y": y
        }

    @staticmethod
    def loss_fn(preds, batch, **kwargs):
        """
        Compute standard cross-entropy loss.
        
        Args:
            preds: Model predictions
            batch: Dict with 'y' (labels)
            **kwargs: Additional parameters (unused)
        """
        y = batch['y']
        return F.cross_entropy(preds, y)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Training loader with LBP
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=lambda b: LocalBinaryPattern.collate_train_step_fn(b, apply_lbp=True, radius=1)
    )
    
    # Validation loader with LBP
    val_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda b: LocalBinaryPattern.collate_val_step_fn(b, apply_lbp=True, radius=1)
    )
    
    # In LightningModule:
    # def training_step(self, batch, batch_idx):
    #     x = batch["x"]
    #     preds = self(x)
    #     loss = LocalBinaryPattern.loss_fn(preds, batch)
    #     return loss