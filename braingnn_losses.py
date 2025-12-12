"""
BrainGNN Loss Functions

This module implements loss functions inspired by BrainGNN for graph neural networks:
- TopK Loss: Forces attention scores to be sparse (0 or 1), facilitating interpretation
- Consistency Loss: Enforces that subjects of the same class rely on similar ROIs

These losses help the model learn interpretable attention patterns and consistent
feature selection across similar samples.
"""

import torch
import torch.nn.functional as F

EPS = 1e-10  # Small epsilon to avoid numerical issues in log operations

def topk_loss(s, ratio):
    """
    TopK Pooling Loss (BrainGNN).
    
    Forces the attention scores 's' to be sparse (either 0 or 1), facilitating interpretation.
    This loss encourages the model to focus on a subset of important ROIs.
    
    Mathematical formulation:
    - Top K scores should be close to 1: minimize -log(s_top_k)
    - Bottom K scores should be close to 0: minimize -log(1 - s_bottom_k)
    - Total loss = loss_keep + loss_discard
    
    Args:
        s: Attention scores / ROI weights [Batch, Num_Nodes]
           Should be in range [0, 1] (after softmax or sigmoid)
        ratio: Pooling ratio (0.5 means keep 50% of nodes).
               If > 0.5, it's treated as keeping ratio, so we penalize the bottom (1-ratio).
               Example: ratio=0.5 -> keep top 50%, discard bottom 50%
    
    Returns:
        torch.Tensor: TopK loss value (scalar, requires_grad=True)
                     Returns 0.0 if NaN/Inf detected (with warning)
    """
    # Check for NaN or Inf in input
    if torch.isnan(s).any() or torch.isinf(s).any():
        print(f"[Warning] topk_loss: NaN/Inf detected in input, returning 0")
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    
    if ratio > 0.5:
        # If we want to keep, say, 70% (0.7), we penalize the bottom 30%.
        # BrainGNN logic seems to invert if > 0.5, ensuring we look at the tail properly.
        ratio = 1 - ratio
    
    # Ensure ratio is in [0, 1]
    ratio = max(0.0, min(1.0, ratio))
    
    # Sort scores to identify top-k and bottom-k
    s_sorted = s.sort(dim=1).values
    
    # Calculate number of nodes to penalize
    num_nodes = s.size(1)
    k = max(1, int(num_nodes * ratio))  # Ensure k >= 1
    k = min(k, num_nodes - 1)  # Ensure k < num_nodes to avoid empty slice
    
    # Check if k is valid
    if k <= 0 or k >= num_nodes:
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    
    # Penalize:
    # 1. Top K scores should be close to 1 -> minimize -log(s)
    # 2. Bottom K scores should be close to 0 -> minimize -log(1-s)
    # Note: s_sorted[:,-k:] are the top k (largest values)
    #       s_sorted[:,:k] are the bottom k (smallest values)
    
    # Clamp values to avoid log(0) even with EPS
    s_sorted = torch.clamp(s_sorted, min=EPS, max=1-EPS)
    
    # Check for NaN after clamp
    if torch.isnan(s_sorted).any():
        print(f"[Warning] topk_loss: NaN after clamp, returning 0")
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    
    # Calculate losses with safety checks
    top_k_scores = s_sorted[:, -k:]  # Top k scores
    bottom_k_scores = s_sorted[:, :k]  # Bottom k scores
    
    # Check if slices are not empty
    if top_k_scores.numel() == 0 or bottom_k_scores.numel() == 0:
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    
    loss_keep = -torch.log(top_k_scores).mean()
    loss_discard = -torch.log(1 - bottom_k_scores).mean()
    
    # Check if losses are valid
    if torch.isnan(loss_keep) or torch.isinf(loss_keep):
        loss_keep = torch.tensor(0.0, device=s.device, requires_grad=True)
    if torch.isnan(loss_discard) or torch.isinf(loss_discard):
        loss_discard = torch.tensor(0.0, device=s.device, requires_grad=True)
    
    total_loss = loss_keep + loss_discard
    
    # Final check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"[Warning] topk_loss: Final loss is NaN/Inf, returning 0")
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    
    return total_loss


def consist_loss(s, device):
    """
    Group Level Consistency Loss (BrainGNN).
    
    Enforces that subjects of the same class rely on similar ROIs by penalizing
    variance in attention scores across subjects in the same group.
    
    Mathematical formulation:
    - W: Similarity matrix (all ones = force all subjects to be similar)
    - L: Graph Laplacian = D - W (where D is degree matrix)
    - Loss = Trace(S^T * L * S) / N^2
      This measures the variance of attention scores across subjects.
      Lower loss = more consistent ROI selection across subjects.
    
    Args:
        s: Attention scores for a group of subjects [Batch_Group, Num_Nodes]
           Each row represents one subject's attention scores over ROIs
        device: Torch device for tensor operations
    
    Returns:
        torch.Tensor: Consistency loss value (scalar)
                     Returns 0 if input is empty
    """
    if len(s) == 0:
        return 0
    
    # Ensure scores are normalized (BrainGNN uses sigmoid, we might use softmax or sigmoid in model)
    # Assuming 's' is already in [0, 1] range.
    s = torch.sigmoid(s) if s.max() > 1.0 else s
    
    # W: Similarity matrix between subjects (All ones = force everyone to be similar)
    W = torch.ones(s.shape[0], s.shape[0], device=device)
    
    # D: Degree matrix (diagonal matrix with row sums of W)
    D = torch.eye(s.shape[0], device=device) * torch.sum(W, dim=1)
    
    # L: Graph Laplacian
    L = D - W
    
    # Loss = Trace(S^T * L * S) / N^2
    # Measures the variance of scores across subjects
    # Lower values indicate more consistent ROI selection
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    
    return res
