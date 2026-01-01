import torch
import numpy as np
import cv2
import torch.nn.functional as F
from improvements.improvement_method import ImprovementMethod

class HOGConcatenation(ImprovementMethod):
    """
    HOG (Histogram of Oriented Gradients) feature extraction and concatenation.
    This method extracts HOG features from images in a batch and prepares them 
    to be concatenated either as additional spatial channels or as a flat vector.
    """

    @staticmethod
    def _extract_hog_batch(images: torch.Tensor, hog_params: dict) -> torch.Tensor:
        """
        Helper to extract HOG features for a batch of images using OpenCV.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            hog_params: Configuration for OpenCV HOGDescriptor
        """
        batch_size = images.shape[0]
        hog_features_list = []
        
        # Standard HOG parameters
        win_size = hog_params.get('win_size', (64, 128)) # Width, Height
        block_size = hog_params.get('block_size', (16, 16))
        block_stride = hog_params.get('block_stride', (8, 8))
        cell_size = hog_params.get('cell_size', (8, 8))
        nbins = hog_params.get('nbins', 9)

        hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )

        for i in range(batch_size):
            # Convert PyTorch (C, H, W) to OpenCV (H, W, C) uint8
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            
            # Normalize to 0-255 if necessary
            if img_np.max() <= 1.01:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # Convert to Grayscale
            if img_np.shape[2] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            # Resize to win_size for consistent descriptor length
            gray_resized = cv2.resize(gray, win_size)

            # Compute HOG
            descriptor = hog.compute(gray_resized)
            hog_features_list.append(torch.from_numpy(descriptor).flatten())

        return torch.stack(hog_features_list)

    @staticmethod
    def collate_train_step_fn(batch, **kwargs):
        """
        Extracts HOG features and returns them in the batch dictionary.
        
        Args:
            batch: List of (image, label) tuples
            **kwargs: 
                - win_size: (w, h) for HOG extraction (default: 64x128)
                - nbins: number of orientation bins (default: 9)
        """
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        # Extract HOG features
        hog_features = HOGConcatenation._extract_hog_batch(x, kwargs)
        
        return {
            "x": x,
            "hog": hog_features,
            "y": y
        }

    @staticmethod
    def collate_val_step_fn(batch, **kwargs):
        """Standard validation processing with HOG features included."""
        return HOGConcatenation.collate_train_step_fn(batch, **kwargs)

    @staticmethod
    def loss_fn(preds, batch, **kwargs):
        """
        Standard Cross Entropy loss. HOG concatenation is a feature-level 
        improvement and does not typically modify the loss function structure.
        """
        y = batch['y']
        return F.cross_entropy(preds, y)

if __name__ == "__main__":
    # Example usage with dummy data
    dummy_batch = [(torch.rand(3, 224, 224), 0) for _ in range(4)]
    
    # Simulate DataLoader call
    processed_batch = HOGConcatenation.collate_train_step_fn(
        dummy_batch, 
        win_size=(64, 64), 
        nbins=9
    )
    
    print(f"Image batch shape: {processed_batch['x'].shape}")
    print(f"HOG features shape: {processed_batch['hog'].shape}")
    print(f"Labels shape: {processed_batch['y'].shape}")

