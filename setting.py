"""
Configuration and argument parsing for BrainPromptGAT.

This module defines all command-line arguments and hyperparameters for the
BrainPromptGAT framework, including model architecture, training settings,
and loss function parameters.
"""

import argparse


def get_args():
    """
    Parse command-line arguments for BrainPromptGAT.
    
    Returns:
        argparse.Namespace: Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="BrainPromptGAT - GAT-based Domain Adaptation")
    parser.add_argument("--gpu", type=str, default=0, help='GPU device number')
    parser.add_argument("--seed", type=int, default=123, help='Random seed for reproducibility')

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Physical batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, 
                        help="Gradient accumulation steps (effective batch_size = batch_size * accumulation_steps)")
    parser.add_argument("--kFold", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--epoch_cf", type=int, default=200, help="Maximum number of training epochs")
    
    # Model saving
    parser.add_argument("--best_epoch", type=int, default=25, 
                        help="Epoch at which to save model (if save_model=1)")
    parser.add_argument("--save_model", type=int, default=0, 
                        help="0: don't save, 1: save model at best_epoch")
    parser.add_argument("--use_optuna", type=int, default=0, 
                        help="1 to use Optuna for hyperparameter optimization, 0 for single run")

    # Model selection
    parser.add_argument('--model', type=str, 
                        choices=['gnn_target_model', 'gnn_source_model1', 'gnn_source_model2', 
                                'gnn_source_model12', 'gnn_baseline_model'], 
                        default='gnn_baseline_model',
                        help="Choose the model to use")
    
    # Dataset selection
    parser.add_argument('--site', type=str, choices=['NYU', 'UCLA', 'UM', 'USM'], default='NYU',
                        help="Choose the target site for evaluation")
    
    # Number of source domains for prompt fusion (only for target_model)
    parser.add_argument('--num_sources', type=int, choices=[1, 3, 6, 10, 12, 15], default=1,
                        help="Number of source domains to use for prompt fusion")
    
    # Source domain for source prompt training
    parser.add_argument('--site_source', type=str, 
                        choices=['NYU', 'UCLA', 'UM', 'USM', 'KKI', 'Leuven', 'MaxMun', 'Pitt', 
                                'Trinity', 'Yale', 'Caltech', 'CMU', 'Olin', 'SBL', 'SDSU', 'Stanford'], 
                        default='CMU',
                        help="Source site for prompt training")
    
    # Basic hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate (increased from original for better learning)")
    parser.add_argument("--decay", type=float, default=0.0, 
                        help="Weight decay (set to 0.0 to avoid erasing weights)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--temperature', type=int, default=8, 
                        help="Temperature parameter for prompt fusion (target model)")
    
    # Target domain learning rates (for future target model implementation)
    parser.add_argument("--prompt_lr", type=float, default=0.0000851, 
                        help="Learning rate for prompt parameters")
    parser.add_argument("--model_lr", type=float, default=0.0001, 
                        help="Learning rate for model parameters")
    
    # GAT-specific hyperparameters
    parser.add_argument("--gat_heads", type=int, default=4, 
                        help="Number of attention heads in first GAT layer")
    parser.add_argument("--gat_hidden_dim", type=int, default=32, 
                        help="Hidden dimension for GAT layers")
    parser.add_argument("--gat_num_layers", type=int, default=2, 
                        help="Number of GAT layers")
    parser.add_argument("--temporal_strategy", type=str, choices=['mean', 'separate'], default='mean',
                        help="Strategy for handling temporal windows: 'mean' (average) or 'separate' (process separately)")
    parser.add_argument("--edge_threshold", type=float, default=0.0,
                        help="Threshold for sparsifying graphs. 0.0 = dense graph (keep all connections)")
    
    # Regularization
    parser.add_argument("--l1_weight", type=float, default=0.01,
                        help="Weight for L1 regularization on soft prompt")
    parser.add_argument("--warmup_epochs", type=int, default=50,
                        help="Number of epochs to linearly ramp up L1 regularization")
    
    # Learning rate scheduler (as in BrainGNN_Pytorch)
    parser.add_argument("--scheduler_step_size", type=int, default=20,
                        help="Step size for StepLR scheduler (not used, we use ReduceLROnPlateau)")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5,
                        help="Gamma (decay factor) for StepLR scheduler (not used)")
    
    # Early stopping (as in gat-li)
    parser.add_argument("--early_stopping_patience", type=int, default=30,
                        help="Number of epochs to wait before early stopping")
    
    # Loss function parameters
    parser.add_argument("--focal_gamma", type=float, default=1.0,
                        help="Gamma parameter for Focal Loss (0.0=CE, >0.0=Focal, higher=focus on hard examples)")
    parser.add_argument("--lambda_tpk", type=float, default=0.1,
                        help="Weight for TopK Loss (interpretable regularization, encourages sparse attention)")

    args = parser.parse_args()
    return args
