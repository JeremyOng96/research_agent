### Improvement Code
import torch
import numpy as np
from typing import Tuple, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

def mixup_collate_fn(batch, alpha=1.0):
    """
    Collate function that applies Mixup to a batch.
    """
    # 1. Standard collation: stack list of (x, y) into tensors
    # This turns [(img1, lbl1), (img2, lbl2)] -> [Batch_Images, Batch_Labels]
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # 2. Apply your Mixup logic
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    # 3. Return a dictionary (cleanest for Lightning)
    return {
        "x": mixed_x,
        "y_a": y_a,
        "y_b": y_b,
        "lam": lam
    }

if __name__ == "__main__":
    # Example usage with LightningModule
    class MixupModel(L.LightningModule):
        def training_step(self, batch, batch_idx):
            x = batch["x"]
            y_a, y_b = batch["y_a"], batch["y_b"]
            lam = batch["lam"]
            
            preds = self(x)
            
            # Standard Mixup Loss calculation
            loss = lam * F.cross_entropy(preds, y_a) + (1 - lam) * F.cross_entropy(preds, y_b)
            
            self.log("train_loss", loss)
            return loss

    # In your DataModule or setup:
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=lambda b: mixup_collate_fn(b, alpha=0.4)
    )