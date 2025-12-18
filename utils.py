"""
Utility functions for BrainPromptGAT.

This module provides helper functions for:
- Model saving and loading
- Random seed setup for reproducibility
- Results saving to Excel format
- Parameter counting
- Demographic data loading
"""

import os
import torch
import numpy as np
import random
import xlwt
import time
import json


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


def load_demographic_data(site, base_dir=None):
    """
    Load demographic data (age, sex, site_id) for a given site.
    
    Args:
        site: Site name (e.g., 'NYU', 'UCLA', 'UM', 'USM')
        base_dir: Base directory for data. If None, tries to find automatically.
    
    Returns:
        tuple: (demographics_dict, index_to_file_id_list)
               - demographics_dict: Mapping from FILE_ID to demographic info
               - index_to_file_id_list: List mapping data index to FILE_ID
               Returns (empty_dict, empty_list) if files not found.
    """
    if base_dir is None:
        # Try to find data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Try BrainPrompt directory first
        demographics_path_brainprompt = os.path.join(
            parent_dir, 'BrainPrompt', 'data', 'correlation', site, 
            f'{site}_15_site_demographics.json'
        )
        
        # Try root data directory
        demographics_path_root = os.path.join(
            parent_dir, 'data', 'correlation', site,
            f'{site}_15_site_demographics.json'
        )
        
        if os.path.exists(demographics_path_brainprompt):
            demographics_path = demographics_path_brainprompt
            index_mapping_path = os.path.join(
                parent_dir, 'BrainPrompt', 'data', 'correlation', site,
                f'{site}_15_site_index_to_file_id.json'
            )
        elif os.path.exists(demographics_path_root):
            demographics_path = demographics_path_root
            index_mapping_path = os.path.join(
                parent_dir, 'data', 'correlation', site,
                f'{site}_15_site_index_to_file_id.json'
            )
        else:
            print(f"Warning: Demographic data not found for site {site}")
            print(f"  Tried: {demographics_path_brainprompt}")
            print(f"  Tried: {demographics_path_root}")
            return {}, []
    else:
        demographics_path = os.path.join(
            base_dir, 'data', 'correlation', site,
            f'{site}_15_site_demographics.json'
        )
        index_mapping_path = os.path.join(
            base_dir, 'data', 'correlation', site,
            f'{site}_15_site_index_to_file_id.json'
        )
    
    if not os.path.exists(demographics_path):
        print(f"Warning: Demographic data file not found: {demographics_path}")
        return {}, []
    
    try:
        with open(demographics_path, 'r', encoding='utf-8') as f:
            demographics = json.load(f)
        print(f"Loaded demographic data for {len(demographics)} subjects from {demographics_path}")
        
        # Load index mapping if available
        index_to_file_id = []
        if os.path.exists(index_mapping_path):
            with open(index_mapping_path, 'r', encoding='utf-8') as f:
                index_to_file_id = json.load(f)
            print(f"Loaded index mapping for {len(index_to_file_id)} samples from {index_mapping_path}")
        else:
            print(f"Warning: Index mapping file not found: {index_mapping_path}")
        
        return demographics, index_to_file_id
    except Exception as e:
        print(f"Error loading demographic data: {e}")
        return {}, []

