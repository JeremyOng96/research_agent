"""
Custom Gabor filter transform for MMPretrain data pipeline.

This can be added to the training pipeline to apply Gabor filtering 
as data augmentation or feature enhancement.
"""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GaborFilter(BaseTransform):
    """Apply Gabor filter to images with optional probability.
    
    This single class handles both deterministic filtering (prob=1.0) and 
    random augmentation (prob<1.0).
    
    Args:
        prob (float): Probability of applying Gabor filter. Default: 1.0 (always apply)
        ksize (int): Size of the Gabor kernel. Default: 21
        sigma (float): Standard deviation of the Gaussian envelope. Default: 5.0
        theta (float): Orientation of the normal to the parallel stripes. Default: 0
        lambd (float): Wavelength of the sinusoidal factor. Default: 10.0
        gamma (float): Spatial aspect ratio. Default: 0.5
        psi (float): Phase offset. Default: 0
        num_orientations (int): Number of orientations to apply. Default: 4
        concatenate (bool): If True, concatenate filtered images as channels.
                          If False, average them. Default: False
    
    Examples:
        >>> # Always apply Gabor filter
        >>> dict(type='GaborFilter', prob=1.0, num_orientations=4)
        
        >>> # Apply Gabor filter with 50% probability (augmentation)
        >>> dict(type='GaborFilter', prob=0.5, num_orientations=4)
        
        >>> # Concatenate as additional features
        >>> dict(type='GaborFilter', concatenate=True, num_orientations=8)
    """
    
    def __init__(self, 
                 prob=1.0,
                 ksize=21,
                 sigma=5.0,
                 theta=0,
                 lambd=10.0,
                 gamma=0.5,
                 psi=0,
                 num_orientations=4,
                 concatenate=False):
        super().__init__()
        self.prob = prob
        self.ksize = ksize
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.num_orientations = num_orientations
        self.concatenate = concatenate
    
    def _apply_gabor(self, img, theta):
        """Apply Gabor filter with specific orientation."""
        kernel = cv2.getGaborKernel(
            (self.ksize, self.ksize),
            self.sigma,
            theta,
            self.lambd,
            self.gamma,
            self.psi,
            ktype=cv2.CV_32F
        )
        
        # Apply filter to each channel
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        return filtered
    
    def transform(self, results):
        """Apply Gabor filters to the image with probability.
        
        Args:
            results (dict): Result dict containing the image data.
            
        Returns:
            dict: Result dict with potentially Gabor filtered image.
        """
        # Check probability - skip if random value exceeds prob
        if np.random.rand() >= self.prob:
            return results
        
        img = results['img']
        
        # Generate orientations
        orientations = [
            np.pi * i / self.num_orientations 
            for i in range(self.num_orientations)
        ]
        
        # Apply Gabor filters with different orientations
        filtered_images = []
        for theta in orientations:
            filtered = self._apply_gabor(img, theta)
            filtered_images.append(filtered)
        
        if self.concatenate:
            # Stack all filtered images as additional channels
            # This will increase the number of channels
            filtered_stack = np.stack(filtered_images, axis=-1)
            # Concatenate with original image
            results['img'] = np.concatenate([img, filtered_stack], axis=-1)
        else:
            # Average all filtered images and replace original
            results['img'] = np.mean(filtered_images, axis=0).astype(img.dtype)
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'ksize={self.ksize}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'lambd={self.lambd}, '
        repr_str += f'gamma={self.gamma}, '
        repr_str += f'num_orientations={self.num_orientations}, '
        repr_str += f'concatenate={self.concatenate})'
        return repr_str

