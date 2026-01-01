import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from improvements.improvement_method import ImprovementMethod


class Mixup(ImprovementMethod):
    """Mixup data augmentation method."""
    
    @staticmethod
    def collate_train_step_fn(batch, **kwargs):
        """Apply Mixup to training batch."""
        alpha = kwargs.get('alpha', 1.0)
        
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return {
            "x": mixed_x,
            "y_a": y_a,
            "y_b": y_b,
            "lam": lam
        }

    @staticmethod
    def collate_val_step_fn(batch, **kwargs):
        """Validation doesn't use Mixup."""
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        return {
            "x": x,
            "y": y
        }

    @staticmethod
    def loss_fn(preds, batch, **kwargs):
        """
        Compute Mixup loss.
        
        Args:
            preds: Model predictions
            batch: Dict with 'y_a', 'y_b', 'lam' (or just 'y' for validation)
        """
        # Check if it's a mixup batch (has y_a, y_b, lam)
        if 'y_a' in batch and 'y_b' in batch and 'lam' in batch:
            # Training with mixup
            y_a = batch['y_a']
            y_b = batch['y_b']
            lam = batch['lam']
            return lam * F.cross_entropy(preds, y_a) + (1 - lam) * F.cross_entropy(preds, y_b)
        else:
            # Validation without mixup
            y = batch['y']
            return F.cross_entropy(preds, y)


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader
    
    # Training loader with Mixup
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=lambda b: Mixup.collate_train_step_fn(b, alpha=0.4)
    )
    
    # Validation loader
    val_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda b: Mixup.collate_val_step_fn(b)
    )
