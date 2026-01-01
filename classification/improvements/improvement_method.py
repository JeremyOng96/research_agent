from abc import ABC, abstractmethod

class ImprovementMethod(ABC):
    """Abstract base class for data improvement methods."""
    
    @staticmethod
    @abstractmethod
    def collate_train_step_fn(batch, **kwargs):
        """
        Collate function for training batches.
        
        Args:
            batch: List of (image, label) tuples
            **kwargs: Method-specific parameters
            
        Returns:
            Dict with processed batch data
        """
        pass

    @staticmethod
    @abstractmethod
    def collate_val_step_fn(batch, **kwargs):
        """
        Collate function for validation batches.
        
        Args:
            batch: List of (image, label) tuples
            **kwargs: Method-specific parameters
            
        Returns:
            Dict with processed batch data
        """
        pass

    @staticmethod
    @abstractmethod
    def loss_fn(preds, batch, **kwargs):
        """
        Compute loss for the improvement method.
        
        Args:
            preds: Model predictions
            batch: Batch dict from collate function
            **kwargs: Additional loss parameters
            
        Returns:
            Loss tensor
        """
        pass