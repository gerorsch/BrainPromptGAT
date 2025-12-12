"""
Focal Loss (multi-class) utility for BrainPromptGAT.

Focal Loss addresses class imbalance by down-weighting easy examples and
focusing training on hard examples. This is particularly useful for
medical imaging datasets where classes may be imbalanced.

Adapted from implementation in focal_loss/Focal-loss-PyTorch/focal_loss.py

Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t: Probability of the true class
    - gamma: Focusing parameter (higher = more focus on hard examples)
    - alpha: Class balancing weights (optional)
    
    The (1 - p_t)^gamma term down-weights easy examples (high p_t),
    allowing the model to focus on hard examples (low p_t).
    
    Args:
        gamma (float): Focusing parameter. Default: 2.0
                      Higher values focus more on hard examples
        alpha (list or tensor, optional): Class balancing weights [num_classes]
                                         If None, no class balancing
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
        num_classes (int): Number of classes. Required if alpha is provided
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', num_classes=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.
        
        Args:
            inputs: Classification logits [Batch, num_classes]
            targets: Ground truth labels [Batch] (long tensor)
        
        Returns:
            torch.Tensor: Focal loss value (scalar if reduction='mean' or 'sum')
        """
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Standard cross-entropy loss
        ce_loss = -targets_one_hot * torch.log(probs + 1e-12)
        
        # Probability of true class for each sample
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        # Higher weight for hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class balancing weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Final focal loss
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
