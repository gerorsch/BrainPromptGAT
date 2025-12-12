"""
Utility functions for BrainPromptGAT.

This module provides helper functions for:
- Model saving and loading
- Random seed setup for reproducibility
- Results saving to Excel format
- Parameter counting
"""

import os
import torch
import numpy as np
import random
import xlwt
import time


def save_model_with_checks(model, save_path):
    """
    Save model state dictionary with directory creation.
    
    Args:
        model: PyTorch model to save
        save_path: Path where to save the model
    """
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def setup_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for PyTorch (CPU and CUDA), NumPy, and Python random module.
    Also sets CuDNN to deterministic mode.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_xlsx(ts_result, model, site, max_epoch, lr, batch_size, weight_decay, num_sources=None):
    """
    Save cross-validation results to Excel file.
    
    Saves results from 5-fold cross-validation including metrics for each fold
    and average across folds.
    
    Args:
        ts_result: List of results for each fold
        model: Model name
        site: Site name
        max_epoch: Best epoch number
        lr: Learning rate used
        batch_size: Batch size used
        weight_decay: Weight decay used
        num_sources: Number of source domains (optional, for future use)
    """
    if num_sources is None:
        num_sources = 1
    
    runtime_id = './result/{}/{}/{}-{}_site_result-{}-{}-{}-{}-{}'.format(
        site, num_sources, model, site, max_epoch, lr, batch_size, weight_decay,
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    )
    save_dir = os.path.dirname(runtime_id)
    os.makedirs(save_dir, exist_ok=True)

    f = xlwt.Workbook('encoding = utf-8')
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    # Calculate average across 5 folds
    a = np.average([ts_result[1], ts_result[2], ts_result[3], ts_result[4], ts_result[5]], axis=0).tolist()
    a[0] = 'average'
    ts_result.append(a)
    for j in range(len(ts_result)):
        for i in range(len(ts_result[j])):
            sheet1.write(j, i, ts_result[j][i])
    f.save(runtime_id + '.xlsx')


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
               - total_params: Total number of parameters
               - trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

