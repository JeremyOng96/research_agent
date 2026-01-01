import torch
import torch.nn.functional as F
from typing import List, Tuple, Union

class LocalBinaryPatternBatch:
    """
    Implements Local Binary Pattern (LBP) as a batch processing step.
    This is designed to be used within a collate_fn or as a post-processing 
    step in a PyTorch DataLoader pipeline.
    
    Attributes:
        p (int): Number of circularly symmetric neighbor set points (default: 8).
        r (int): Radius of circle (spatial resolution of the operator) (default: 1).
    """

    def __init__(self, radius: int = 1):
        self.radius = radius
        # Define the 8 neighbors for a standard 3x3 LBP (Radius 1)
        # Order: Top-Left, Top, Top-Right, Right, Bottom-Right, Bottom, Bottom-Left, Left
        self.offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),   (1, 1),  (1, 0),
            (1, -1),  (0, -1)
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Batch of images of shape (B, C, H, W).
                             Values are expected to be in range [0, 1] or [0, 255].
        
        Returns:
            torch.Tensor: LBP transformed batch of shape (B, C, H, W).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")

        b, c, h, w = x.shape
        # Pad the input to handle edges based on radius
        padded = F.pad(x, (self.radius, self.radius, self.radius, self.radius), mode='reflect')
        
        # Initialize LBP output
        lbp_output = torch.zeros_like(x, dtype=torch.float32)

        # Iterate through the 8 neighbors
        for i, (dy, dx) in enumerate(self.offsets):
            # Calculate the binary weight (2^i)
            weight = 2**i
            
            # Extract the shifted neighbor window
            # We slice the padded image to get the same spatial size as original x
            # Start indices based on offset relative to (radius, radius)
            y_start = self.radius + dy
            x_start = self.radius + dx
            
            neighbor_window = padded[:, :, y_start:y_start + h, x_start:x_start + w]
            
            # Comparison: 1 if neighbor >= center, else 0
            # We use .float() to allow multiplication with weight
            mask = (neighbor_window >= x).float()
            
            # Accumulate the weighted sum
            lbp_output += mask * weight

        return lbp_output

def lbp_collate_fn(batch: List[Tuple[torch.Tensor, Union[int, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate_fn that applies LBP transformation to the batch.
    
    Args:
        batch: List of tuples (image_tensor, label)
        
    Returns:
        Tuple of (lbp_image_batch, label_batch)
    """
    images, labels = zip(*batch)
    
    # Stack images into (B, C, H, W)
    image_batch = torch.stack(images, dim=0)
    
    # Stack labels
    if isinstance(labels[0], torch.Tensor):
        label_batch = torch.stack(labels, dim=0)
    else:
        label_batch = torch.tensor(labels)

    # Apply LBP
    # Note: LBP is typically applied to grayscale. 
    # If images are RGB, this applies LBP per channel.
    lbp_transformer = LocalBinaryPatternBatch(radius=1)
    lbp_batch = lbp_transformer(image_batch)
    
    # Normalize LBP values to [0, 1] range if needed (LBP max is 255)
    lbp_batch = lbp_batch / 255.0

    return lbp_batch, label_batch
