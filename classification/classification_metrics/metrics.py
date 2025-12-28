import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score
)

class ClassificationMetrics:
    @staticmethod
    def get_aucroc(y_true, y_pred):
        """
        Returns AUROC score for multiclass classification.
        Args:
            y_true: ground truth labels (integers)
            y_pred: predicted probabilities (softmax output, shape: [N, num_classes])
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # For multiclass, we need multi_class parameter
        return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    @staticmethod
    def get_f1(y_true, y_pred):
        """
        Returns F1 score for multiclass classification.
        Args:
            y_true: ground truth labels
            y_pred: predicted class labels (integers, NOT probabilities!)
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # If y_pred is 2D (probabilities), convert to class labels
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # ✅ Add average='macro' for multiclass
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def get_accuracy(y_true, y_pred):
        """Returns accuracy score."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def get_precision(y_true, y_pred):
        """Returns precision score."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        return precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    @staticmethod
    def get_recall(y_true, y_pred):
        """Returns recall score."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # ✅ Add average='macro' for multiclass
        return recall_score(y_true, y_pred, average='macro', zero_division=0)