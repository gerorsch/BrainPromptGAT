"""
Main entry point for BrainPromptGAT experiments.

This module provides:
- Single experiment execution with fixed hyperparameters
- Optuna-based hyperparameter optimization
- Data path resolution and loading

Usage:
    python main.py --model gnn_baseline_model --site UCLA --use_optuna 0
    python main.py --model gnn_baseline_model --site UCLA --use_optuna 1  # Optuna optimization
"""

import setting
import optuna
import os
import sys

# Add parent directory to path to access data from BrainPrompt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

args = setting.get_args()

import gnn_baseline_train
# Note: Other training modules (source1, source2, target) will be added as they are implemented
# import gnn_source1_train
# import gnn_source2_train
# import gnn_target_train


def get_data_paths(args):
    """
    Get data file paths for the specified site.
    
    Tries to find data in BrainPrompt directory first, then root data directory.
    Supports both matrix format (preferred) and vector format (fallback).
    
    Args:
        args: Arguments object with 'site' attribute
    
    Returns:
        tuple: (path_list, path_list_mask, path_label)
               - path_list: Path to data matrix file
               - path_list_mask: Path to temporal mask file
               - path_label: Path to labels file
    """
    # Get absolute path to data directory (shared with BrainPrompt)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir_brainprompt = os.path.join(base_dir, 'BrainPrompt', 'data', 'correlation', args.site)
    data_dir_root = os.path.join(base_dir, 'data', 'correlation', args.site)
    
    # Use BrainPrompt directory if it exists, otherwise use root
    if os.path.exists(data_dir_brainprompt):
        data_dir = data_dir_brainprompt
    else:
        data_dir = data_dir_root
    
    # Try matrix format first, fallback to vector format
    path_list_matrix = os.path.join(data_dir, f'{args.site}_15_site_X_matrix.npy')
    path_list_vector = os.path.join(data_dir, f'{args.site}_15_site_X1.npy')
    path_list_mask = os.path.join(data_dir, f'{args.site}_15_site_X1_mask.npy')
    path_label = os.path.join(data_dir, f'{args.site}_15_site_Y1.npy')
    
    # Use matrix path, but load_matrix_data will fallback to vector if needed
    path_list = path_list_matrix
    print(f'Data directory: {data_dir}')
    print(f'Loading data from: {path_list}')
    print(f'Fallback vector path: {path_list_vector}')
    print(f'Files exist: matrix={os.path.exists(path_list_matrix)}, vector={os.path.exists(path_list_vector)}, mask={os.path.exists(path_list_mask)}, label={os.path.exists(path_label)}')
    
    return path_list, path_list_mask, path_label


def run_single_experiment():
    """
    Run a single experiment with fixed hyperparameters.
    
    Returns:
        float: Final accuracy (as percentage)
    """
    print(f"Running single experiment with model: {args.model}")
    print(f"Hyperparameters: Heads={args.gat_heads}, Dim={args.gat_hidden_dim}, Layers={args.gat_num_layers}, LR={args.lr}")
    
    if args.model == 'gnn_baseline_model':
        path_list, path_list_mask, path_label = get_data_paths(args)
        
        acc = gnn_baseline_train.train_and_test_baseline_model(
            args, path_list, path_list_mask, path_label, args.lr, args.decay
        )
        print(f"Final Accuracy: {acc}")
        return acc
    
    # Add other models here as they are implemented
    return 0


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    
    This function is called by Optuna for each trial. It suggests hyperparameters,
    trains the model, and returns the accuracy to maximize.
    
    Args:
        trial: Optuna trial object for hyperparameter suggestions
    
    Returns:
        float: Model accuracy (to be maximized)
    """
    # Clear CUDA memory before each trial
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Hyperparameter suggestions (more conservative values)
    lr = trial.suggest_categorical("lr", [
        0.00001, 0.00002, 0.00003, 0.00004, 0.00006, 0.0000851,
        0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001
        # Removed very high values (0.1+) that may cause instability
    ])
    
    weight_decay = trial.suggest_categorical("weight_decay", [
        0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01
    ])
    
    # GAT-specific hyperparameters (limited to avoid OOM)
    # Optuna doesn't support dynamic distributions, so we use fixed values
    gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4])  # Removed 8
    gat_hidden_dim = trial.suggest_categorical("gat_hidden_dim", [64, 128, 256])
    gat_num_layers = trial.suggest_categorical("gat_num_layers", [1, 2, 3])
    
    # Check if combination is too heavy and skip if necessary
    if gat_hidden_dim == 256 and gat_num_layers == 3:
        # This combination may cause OOM, return low value
        return 0.0
    
    # Target model parameters (for future implementation)
    temperature = trial.suggest_categorical('temperature', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    
    if args.model == 'gnn_baseline_model':
        path_list, path_list_mask, path_label = get_data_paths(args)
        
        # Update args with GAT hyperparameters
        args.gat_heads = gat_heads
        args.gat_hidden_dim = gat_hidden_dim
        args.gat_num_layers = gat_num_layers
        
        # Train baseline model
        try:
            acc = gnn_baseline_train.train_and_test_baseline_model(
                args, path_list, path_list_mask, path_label, lr, weight_decay
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear memory and return low value for Optuna to avoid this configuration
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"OOM error with params: gat_heads={gat_heads}, hidden_dim={gat_hidden_dim}, layers={gat_num_layers}")
                return 0.0  # Return 0 for Optuna to avoid this configuration
            else:
                raise
    
    # Note: Other model types will be added as they are implemented
    # elif args.model == 'gnn_source_model1':
    #     ...
    # elif args.model == 'gnn_target_model':
    #     ...
    
    return acc


if __name__ == '__main__':
    if args.use_optuna == 1:
        print("Starting Optuna optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200)
        print(study.best_params)
    else:
        run_single_experiment()
