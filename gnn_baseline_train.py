"""
Training script for GAT Baseline Model

This module implements the training and evaluation pipeline for the GAT baseline model,
including 5-fold cross-validation, early stopping, learning rate scheduling, and
comprehensive loss functions (Focal Loss, TopK Loss, Diversity Loss, L1 regularization).
"""
from utils import save_model_with_checks, setup_seed, save_xlsx, count_parameters
from focal_loss_utils import FocalLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn import metrics
import math, copy, time
import gnn_baseline_model
import xlwt
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split, KFold
from graph_utils import load_matrix_data, temporal_graphs_to_batch
import warnings
warnings.filterwarnings("ignore")


def make_model_baseline_model(num_nodes=116, input_dim=116, hidden_dim=128, num_heads=2,
                              num_layers=2, dropout=0.5, num_classes=2, temporal_strategy='mean', device=0):
    """
    Create and initialize GAT baseline model.
    
    Args:
        num_nodes (int): Number of nodes (ROIs). Default: 116
        input_dim (int): Input feature dimension. Default: 116
        hidden_dim (int): Hidden dimension for GAT. Default: 128
        num_heads (int): Number of attention heads. Default: 2
        num_layers (int): Number of GAT layers. Default: 2
        dropout (float): Dropout rate. Default: 0.5
        num_classes (int): Number of output classes. Default: 2
        temporal_strategy (str): Temporal aggregation strategy. Default: 'mean'
        device (int): Device ID. Default: 0
    
    Returns:
        GATBaselineModel: Initialized model
    """
    model = gnn_baseline_model.GATBaselineModel(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        temporal_strategy=temporal_strategy
    )
    
    # Initialize parameters (optimized initialization)
    # NOTE: fc1.weight and fc1.bias are already initialized in model.__init__
    # with optimized values, so we skip them to avoid overwriting
    for name, p in model.named_parameters():
        if name == "soft_prompt":
            continue  # Keep fine-tuning at zero (semantic base prompt already projected)
        if name == "fc1.weight" or name == "fc1.bias":
            continue  # Skip fc1 - already initialized in __init__ with optimized values
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)  # Bias initialized to zero
    
    return model


def configure_model_optimizer_baseline_model(args, model, device, lr, decay):
    """
    Configure model parameters and optimizer.
    
    Args:
        args: Arguments object with model configuration
        model: GAT model instance
        device: Torch device (CPU or CUDA)
        lr: Learning rate (can be None for auto-adjustment)
        decay: Weight decay (will be forced to 0.0)
    
    Returns:
        tuple: (model, model_param_group, optimizer)
    """
    if args.model == 'gnn_baseline_model':
        model.to(device)
        
        # Adjust LR more conservatively
        # LR too high can cause instability, too low may not learn
        if lr is None:
            lr = 0.0005  # More conservative default value
            print(f"[Auto-Adjust] LR not specified, using default: {lr}")
        elif lr < 0.0001:
            print(f"[Warning] LR too low ({lr}), may not learn. Consider increasing.")
        elif lr > 0.01:
            print(f"[Warning] LR too high ({lr}), may cause instability. Consider decreasing.")
        
        # Force weight decay = 0 to avoid erasing weights
        decay = 0.0
        print(f"[Auto-Adjust] Weight Decay set to {decay}")
        
        model_param_group = []
        model_param_group.append({"params": model.parameters()})
        optimizer = torch.optim.AdamW(model_param_group, lr=lr, weight_decay=decay)
    
    return model, model_param_group, optimizer


def train_and_test_baseline_model(args, path_data, path_data_mask, path_label, lr, weight_decay):
    """
    Train and evaluate GAT baseline model with 5-fold cross-validation.
    
    This function implements the complete training pipeline including:
    - Data loading and preprocessing
    - 5-fold cross-validation
    - Model training with early stopping
    - Learning rate scheduling
    - Comprehensive evaluation metrics
    
    Args:
        args: Arguments object with all hyperparameters
        path_data: Path to data matrix file (N, 25, 116, 116)
        path_data_mask: Path to temporal mask file
        path_label: Path to labels file
        lr: Learning rate
        weight_decay: Weight decay (will be forced to 0.0)
    
    Returns:
        float: Maximum accuracy across all folds (as percentage)
    """
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load matrix data (N, 25, 116, 116) instead of vector
    X, X_mask = load_matrix_data(path_data, path_data_mask)
    Y = np.load(path_label)
    
    # Convert to numpy for sklearn
    X = X.numpy()
    if X_mask is not None:
        X_mask = X_mask.numpy()
    else:
        # Create mask if not provided (all windows are valid) - shape (N, 25, 25) for compatibility
        X_mask = np.ones((X.shape[0], X.shape[1], X.shape[1]), dtype=bool)
    
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    epoch_list = []
    sen_list = []
    spe_list = []
    loss_all = []
    
    max_acc = 0
    max_precision = 0
    max_recall = 0
    max_f1 = 0
    max_auc = 0
    max_epoch = 0
    
    kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
    kfold_index = 0
    
    for train_index, test_index in kf.split(X):
        kfold_index += 1
        X_train, X_test = X[train_index], X[test_index]
        X_mask_train, X_mask_test = X_mask[train_index], X_mask[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        print('X_train{}'.format(X_train.shape))
        print('X_test{}'.format(X_test.shape))
        print('Y_train{}'.format(Y_train.shape))
        print('Y_test{}'.format(Y_test.shape))
        
        # Create model
        model = make_model_baseline_model(
            num_nodes=116,
            input_dim=116,  # Each node gets its row from correlation matrix
            hidden_dim=args.gat_hidden_dim,
            num_heads=args.gat_heads,
            num_layers=args.gat_num_layers,
            dropout=args.dropout,
            num_classes=2,
            temporal_strategy=args.temporal_strategy,
            device=device
        )
        
        # Clear memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Configure optimizer
        model, model_param_group, optimizer = configure_model_optimizer_baseline_model(
            args, model, device, lr, weight_decay
        )
        
        # Add Learning Rate Scheduler (as in BrainGNN_Pytorch and BrainPrompt Paper)
        # ReduceLROnPlateau: reduces LR when loss stops decreasing
        # patience=15 epochs, factor=0.5 (halves LR), min_lr=1e-6 (minimum limit)
        scheduler_patience = getattr(args, 'early_stopping_patience', 30) // 2 + 5  # Approximately half of early stopping patience
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=15,
            min_lr=1e-6  # Minimum limit to avoid LR too small
        )
        # Note: verbose is not supported in ReduceLROnPlateau, but we do manual logging below
        
        # Create validation set for early stopping (20% of training data)
        # As in BrainGNN_Pytorch which uses train/val/test split
        val_size = int(0.2 * len(X_train))
        val_indices = np.random.choice(len(X_train), size=val_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
        
        X_train_final = X_train[train_indices]
        X_val = X_train[val_indices]
        X_mask_train_final = X_mask_train[train_indices]
        X_mask_val = X_mask_train[val_indices]
        Y_train_final = Y_train[train_indices]
        Y_val = Y_train[val_indices]
        
        # Class weights for focal loss (based on training set of this fold)
        num_class_0 = (Y_train_final == 0).sum()
        num_class_1 = (Y_train_final == 1).sum()
        total_samples = len(Y_train_final)
        weight_0 = total_samples / (2 * num_class_0 + 1e-6)
        weight_1 = total_samples / (2 * num_class_1 + 1e-6)
        focal_alpha = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
        focal_gamma = getattr(args, 'focal_gamma', 2.0)
        focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction='mean', num_classes=2)
        
        # Log class balance
        print(f'\n[CLASS BALANCE] Fold {kfold_index}:')
        print(f'  Train: Class 0={num_class_0} ({num_class_0/total_samples*100:.1f}%), Class 1={num_class_1} ({num_class_1/total_samples*100:.1f}%)')
        print(f'  Focal Loss alpha: [{weight_0:.4f}, {weight_1:.4f}]')
        num_test_0 = (Y_test == 0).sum()
        num_test_1 = (Y_test == 1).sum()
        total_test = len(Y_test)
        print(f'  Test: Class 0={num_test_0} ({num_test_0/total_test*100:.1f}%), Class 1={num_test_1} ({num_test_1/total_test*100:.1f}%)')
        if abs(num_class_0 - num_class_1) / total_samples > 0.2:
            print(f'  [WARNING] Significant class imbalance detected (>20% difference)')
        
        # Early stopping: monitors validation loss (as in gat-li)
        best_val_loss = float('inf')
        patience = getattr(args, 'early_stopping_patience', 30)
        patience_counter = 0
        best_model_state = None
        
    from braingnn_losses import topk_loss

    # Training loop
    for epoch in range(args.epoch_cf):
            current_lr = optimizer.param_groups[0]['lr']
            
            model.train()
            train_loss_all = 0
            samples_processed = 0
            
            optimizer.zero_grad()
            
            # Shuffle batch indices
            perm_indices = np.random.permutation(int(X_train_final.shape[0]))
            
            num_batches = int(X_train_final.shape[0]) // int(args.batch_size)
            
            for i in range(0, num_batches):
                idx_batch = perm_indices[
                    i * int(args.batch_size):min((i + 1) * int(args.batch_size), X_train_final.shape[0])
                ]
                
                train_data_batch = X_train_final[idx_batch]  # (batch, 25, 116, 116)
                train_label_batch = Y_train_final[idx_batch]
                
                # Convert to tensors
                train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch = torch.from_numpy(train_label_batch).long()
                
                # Convert matrices to graph batch
                train_mask_batch = torch.from_numpy(X_mask_train_final[idx_batch]).float().to(device)
                batch_data = temporal_graphs_to_batch(
                    train_data_batch,
                    # Disable pruning for dense graphs like BrainGNN
                    threshold=None,
                    temporal_strategy=args.temporal_strategy,
                    window_mask=train_mask_batch
                )
                batch_data = batch_data.to(device)
                batch_data.y = train_label_batch.to(device)
                
                # --- Scale debug (First batch of first epoch) ---
                if epoch == 0 and i == 0:
                    with torch.no_grad():
                        x_stats = batch_data.x
                        print(f"\n[DEBUG SCALE] Image: Mean={x_stats.mean():.4f}, Std={x_stats.std():.4f}, Max={x_stats.max():.4f}")
                        if model.static_prompt is not None:
                            p_stats = model.prompt_projector(model.static_prompt.to(device))
                            print(f"[DEBUG SCALE] Prompt (Proj): Mean={p_stats.mean():.4f}, Std={p_stats.std():.4f}, Max={p_stats.max():.4f}")
                        print(f"[DEBUG SCALE] Soft Prompt: Mean={model.soft_prompt.mean():.4f}, Std={model.soft_prompt.std():.4f}\n")
                # ----------------------------------------------------------
                
                # Enable debug only in first iteration of first epoch
                debug_mode = (epoch == 0 and i == 0)
                outputs, attention_weights, features_intermediate = model(batch_data, debug=debug_mode)
                
                # Use Focal Loss (multi-class)
                loss = focal_loss_fn(outputs, batch_data.y)
                
                # Add regularization to avoid collapse to one class
                # Penalizes when all predictions are from the same class
                probs_batch = F.softmax(outputs, dim=1)
                # Calculate entropy of predictions (low entropy = always same class)
                entropy = -(probs_batch * torch.log(probs_batch + 1e-10)).sum(dim=1).mean()
                # Minimum entropy for 2 classes is log(2) ≈ 0.693
                # Penalize if entropy < 0.5 (too confident in one class)
                diversity_weight = getattr(args, 'diversity_weight', 0.01)  # Small weight to not dominate
                diversity_loss = diversity_weight * torch.clamp(0.5 - entropy, min=0.0)
                loss = loss + diversity_loss
                
                # BrainGNN Features: TopK Loss (TPK)
                # Forces attention weights (ROI importance) to be 0 or 1
                lambda_tpk = getattr(args, 'lambda_tpk', 0.1)
                loss_tpk = torch.tensor(0.0, device=device, requires_grad=True)
                if attention_weights is not None:
                     # Reshape weights from (N, 1) to (Batch, Num_Nodes)
                     # attention_weights has shape (N, 1) where N is total number of nodes in batch
                     # We need to reshape to (Batch, Num_Nodes) using reshape
                     num_nodes = 116  # Fixed number of nodes per graph
                     batch_size = outputs.shape[0]
                     
                     # Remove extra dimension if exists
                     if attention_weights.dim() > 1:
                         attention_weights = attention_weights.squeeze(-1)  # (N,)
                     
                     # Reshape to (Batch, Num_Nodes)
                     # Use direct reshape since we know each graph has exactly num_nodes nodes
                     if attention_weights.shape[0] == batch_size * num_nodes:
                         weights_reshaped = attention_weights.view(batch_size, num_nodes)  # (Batch, Num_Nodes)
                         
                         # Check for NaN or Inf before calculating loss
                         if torch.isnan(weights_reshaped).any() or torch.isinf(weights_reshaped).any():
                             print(f"[Warning] NaN/Inf detected in attention_weights, skipping TPK loss")
                             loss_tpk = torch.tensor(0.0, device=device, requires_grad=True)
                         else:
                             loss_tpk = topk_loss(weights_reshaped, ratio=0.5)
                             # Check if loss_tpk is NaN
                             if torch.isnan(loss_tpk) or torch.isinf(loss_tpk):
                                 print(f"[Warning] TPK Loss is NaN/Inf, setting to 0")
                                 loss_tpk = torch.tensor(0.0, device=device, requires_grad=True)
                     else:
                         # If shape doesn't match, skip TPK loss
                         print(f"[Warning] Unexpected attention_weights shape: {attention_weights.shape}, expected {(batch_size * num_nodes,)}")
                         loss_tpk = torch.tensor(0.0, device=device, requires_grad=True)
                     
                     # Log TPK loss in first iterations for debug
                     if epoch == 0 and i == 0:
                         print(f"[BrainGNN] TPK Loss: {loss_tpk.item():.4f}")
                
                loss = loss + (lambda_tpk * loss_tpk)
                
                # Calculate current regularization weight (Linear Warmup)
                current_epoch = epoch
                if current_epoch < args.warmup_epochs:
                    current_l1_weight = args.l1_weight * (current_epoch / args.warmup_epochs)
                else:
                    current_l1_weight = args.l1_weight
                
                # Add L1 regularization of mask prompt
                l1_reg = current_l1_weight * model.get_mask_prompt_l1_norm()
                loss = loss + l1_reg
                
                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps
                loss.backward()
                
                # Update weights every accumulation_steps or at the end of epoch
                if (i + 1) % args.accumulation_steps == 0 or (i + 1) == num_batches:
                    # Detailed logging of gradients and outputs (first batch of selected epochs)
                    if (epoch == 0 or epoch % 10 == 0) and i == 0:
                        # Calculate gradient norm
                        total_grad_norm = 0.0
                        param_count = 0
                        grad_norms = []
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param_grad_norm = param.grad.data.norm(2).item()
                                total_grad_norm += param_grad_norm ** 2
                                grad_norms.append((name, param_grad_norm))
                                param_count += 1
                        total_grad_norm = total_grad_norm ** 0.5
                        
                        # Log outputs, intermediate features and loss components
                        with torch.no_grad():
                            outputs_stats = outputs.detach()
                            features_stats = features_intermediate.detach()
                            focal_loss_val = focal_loss_fn(outputs_stats, batch_data.y).item()
                            print(f"\n[DEBUG EPOCH {epoch}] Gradients: norm={total_grad_norm:.6f}, params_with_grad={param_count}")
                            print(f"[DEBUG EPOCH {epoch}] Intermediate features (before fc1): mean={features_stats.mean():.4f}, "
                                  f"std={features_stats.std():.4f}, min={features_stats.min():.4f}, max={features_stats.max():.4f}")
                            print(f"[DEBUG EPOCH {epoch}] Outputs (logits): mean={outputs_stats.mean():.4f}, std={outputs_stats.std():.4f}, "
                                  f"min={outputs_stats.min():.4f}, max={outputs_stats.max():.4f}")
                            # Calculate loss per class for diagnosis
                            with torch.no_grad():
                                class_0_mask = (batch_data.y == 0)
                                class_1_mask = (batch_data.y == 1)
                                if class_0_mask.any():
                                    loss_class_0 = focal_loss_fn(outputs_stats[class_0_mask], batch_data.y[class_0_mask]).item()
                                else:
                                    loss_class_0 = 0.0
                                if class_1_mask.any():
                                    loss_class_1 = focal_loss_fn(outputs_stats[class_1_mask], batch_data.y[class_1_mask]).item()
                                else:
                                    loss_class_1 = 0.0
                            
                            print(f"[DEBUG EPOCH {epoch}] Loss components: focal={focal_loss_val:.4f} "
                                  f"(class0={loss_class_0:.4f}, class1={loss_class_1:.4f}), "
                                  f"tpk={loss_tpk.item():.4f}, l1={l1_reg.item():.4f}, "
                                  f"diversity={diversity_loss.item():.4f}, total={loss.item() * args.accumulation_steps:.4f}")
                            
                            # Log top 5 largest gradients
                            if grad_norms:
                                grad_norms_sorted = sorted(grad_norms, key=lambda x: x[1], reverse=True)[:5]
                                print(f"[DEBUG EPOCH {epoch}] Top 5 grad norms: {[(n.split('.')[-1], f'{v:.6f}') for n, v in grad_norms_sorted]}")
                    
                    # Gradient clipping for stability
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Log if gradient was clipped
                    if (epoch == 0 or epoch % 10 == 0) and i == 0 and grad_norm_before_clip > 1.0:
                        print(f"[DEBUG EPOCH {epoch}] Gradient clipped: {grad_norm_before_clip:.4f} -> 1.0")
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Clear intermediate variables
                del batch_data, outputs
            
            if epoch % 5 == 0:
                # Training accuracy
                count = 0
                acc = 0
                model.eval()
                with torch.no_grad():
                    for i in range(0, int(X_train_final.shape[0]) // int(args.batch_size)):
                        idx_batch = perm_indices[
                            i * int(args.batch_size):min((i + 1) * int(args.batch_size), X_train_final.shape[0])
                        ]
                        train_data_batch = X_train_final[idx_batch]
                        train_label_batch = Y_train_final[idx_batch]
                        
                        train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                        train_label_batch = torch.from_numpy(train_label_batch).long()
                        
                        train_mask_batch = X_mask_train_final[idx_batch]
                        batch_data = temporal_graphs_to_batch(
                            train_data_batch,
                            threshold=None,
                            temporal_strategy=args.temporal_strategy,
                            window_mask=train_mask_batch
                        )
                        batch_data = batch_data.to(device)
                        
                        outputs, _, _ = model(batch_data, debug=False)
                        _, indices = torch.max(outputs, dim=1)
                        preds = indices.cpu()
                        acc += metrics.accuracy_score(preds, train_label_batch)
                        count = count + 1
                        
                        del batch_data, outputs
                
                print('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f\tlr: %.6f' % (epoch, loss.item(), acc / count, current_lr))
                loss_all.append(loss.data.item())
            
            # Calculate validation loss for early stopping (as in gat-li)
            # Process validation in batches to avoid OOM (CUDA Out of Memory)
            if epoch % 5 == 0:
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                
                # Create batch indices for validation
                val_num_samples = len(X_val)
                val_batch_size = args.batch_size  # Use same batch size as training for safety
                val_num_batches = (val_num_samples + val_batch_size - 1) // val_batch_size
                
                with torch.no_grad():
                    for i in range(val_num_batches):
                        start_idx = i * val_batch_size
                        end_idx = min((i + 1) * val_batch_size, val_num_samples)
                        
                        # Get slice of current batch
                        # Don't move everything to GPU at once!
                        batch_val_X = X_val[start_idx:end_idx]
                        batch_val_Y = Y_val[start_idx:end_idx]
                        batch_val_Mask = X_mask_val[start_idx:end_idx]
                        
                        val_data_batch = torch.from_numpy(batch_val_X).float().to(device)
                        val_mask_batch = torch.from_numpy(batch_val_Mask).float().to(device)
                        
                        batch_data_val = temporal_graphs_to_batch(
                            val_data_batch,
                            threshold=None,
                            temporal_strategy=args.temporal_strategy,
                            window_mask=val_mask_batch
                        )
                        batch_data_val = batch_data_val.to(device)
                        batch_data_val.y = torch.from_numpy(batch_val_Y).long().to(device)
                        
                        outputs_val, _, _ = model(batch_data_val, debug=False)
                        val_loss = focal_loss_fn(outputs_val, batch_data_val.y)
                        if val_loss.dim() == 0:
                            val_loss = val_loss * (end_idx - start_idx)
                        
                        val_loss_sum += val_loss.item()
                        val_count += (end_idx - start_idx)
                        
                        # Clear memory immediately
                        del val_data_batch, val_mask_batch, batch_data_val, outputs_val
                        torch.cuda.empty_cache()
                
                # Average loss
                avg_val_loss = val_loss_sum / max(val_count, 1)
                
                # Update learning rate scheduler based on validation loss
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f'  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}')
                
                # Early stopping: if validation loss doesn't improve for 'patience' epochs
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f'  Validation loss improved: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'  Early stopping at epoch {epoch} (patience={patience})')
                        # Load best model
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
            
            if epoch % 5 == 0:
                # Testing
                model.eval()
                test_data_batch = torch.from_numpy(X_test).float().to(device)
                
                test_mask_batch = X_mask_test
                batch_data = temporal_graphs_to_batch(
                    test_data_batch,
                    threshold=None,
                    temporal_strategy=args.temporal_strategy,
                    window_mask=test_mask_batch
                )
                batch_data = batch_data.to(device)
                
                outputs, _, features_intermediate = model(batch_data, debug=False)
                
                # Clear memory after inference
                del batch_data
                
                # Use softmax to get probabilities and then argmax
                probs = F.softmax(outputs, dim=1)
                _, indices = torch.max(probs, dim=1)
                pre = indices.cpu().numpy()
                label_test = Y_test
                
                # Debug: check predictions and probabilities distribution
                if epoch == 0 or epoch % 10 == 0:
                    unique, counts = np.unique(pre, return_counts=True)
                    probs_np = probs.detach().cpu().numpy()
                    features_np = features_intermediate.detach().cpu().numpy()
                    # Probability statistics
                    prob_mean = probs_np.mean(axis=0)
                    prob_std = probs_np.std(axis=0)
                    prob_min = probs_np.min(axis=0)
                    prob_max = probs_np.max(axis=0)
                    # Average difference between the two classes
                    prob_diff = np.abs(probs_np[:, 0] - probs_np[:, 1]).mean()
                    print(f'  Debug: Predictions distribution: {dict(zip(unique, counts))}')
                    print(f'  Debug: Intermediate features - mean={features_np.mean():.4f}, std={features_np.std():.4f}, '
                          f'min={features_np.min():.4f}, max={features_np.max():.4f}')
                    print(f'  Debug: Output range: [{outputs.min():.3f}, {outputs.max():.3f}]')
                    print(f'  Debug: Probabilities - Mean: {prob_mean}, Std: {prob_std}, Min: {prob_min}, Max: {prob_max}')
                    print(f'  Debug: Average prob difference between classes: {prob_diff:.4f} (should be > 0.1 for confident predictions)')
                
                acc_test = metrics.accuracy_score(label_test, pre)
                fpr, tpr, _ = metrics.roc_curve(label_test, pre)
                auc = metrics.auc(fpr, tpr)
                tn, fp, fn, tp = metrics.confusion_matrix(label_test, pre).ravel()
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                precision = metrics.precision_score(label_test, pre, zero_division=1)
                f1 = metrics.f1_score(label_test, pre)
                recall = metrics.recall_score(label_test, pre)
                
                acc_list.append(acc_test)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                auc_list.append(auc)
                epoch_list.append(epoch)
                sen_list.append(sen)
                spe_list.append(spe)
                
                print('test result',
                      [kfold_index, epoch, round(acc_test, 4), round(precision, 4), 
                       round(recall, 4), round(f1, 4), round(auc, 4), round(sen, 4), round(spe, 4)])
            
            if epoch == args.best_epoch and args.save_model == 1:
                if args.model == 'gnn_baseline_model':
                    save_path = f'./gnn_baseline_model/{args.site}/15_site/{kfold_index}.pt'
                    save_model_with_checks(model, save_path)
    
    # Calculate best results across all folds
    ts_result = [['kfold_index', 'prec', 'recall', 'acc', 'F1', 'auc', 'sen', 'spe']]
    for i in range(5):
        ts_result.append([])
    
    num = len(acc_list) // 5
    for i in range(num):
        if max_acc < (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + 
                     acc_list[i + 3 * num] + acc_list[i + 4 * num]) / 5:
            max_acc = (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + 
                      acc_list[i + 3 * num] + acc_list[i + 4 * num]) / 5
            max_precision = (precision_list[i] + precision_list[i + num] + 
                            precision_list[i + 2 * num] + precision_list[i + 3 * num] + 
                            precision_list[i + 4 * num]) / 5
            max_recall = (recall_list[i] + recall_list[i + num] + recall_list[i + 2 * num] + 
                         recall_list[i + 3 * num] + recall_list[i + 4 * num]) / 5
            max_f1 = (f1_list[i] + f1_list[i + num] + f1_list[i + 2 * num] + 
                     f1_list[i + 3 * num] + f1_list[i + 4 * num]) / 5
            max_auc = (auc_list[i] + auc_list[i + num] + auc_list[i + 2 * num] + 
                      auc_list[i + 3 * num] + auc_list[i + 4 * num]) / 5
            max_sen = (sen_list[i] + sen_list[i + num] + sen_list[i + 2 * num] + 
                      sen_list[i + 3 * num] + sen_list[i + 4 * num]) / 5
            max_spe = (spe_list[i] + spe_list[i + num] + spe_list[i + 2 * num] + 
                      spe_list[i + 3 * num] + spe_list[i + 4 * num]) / 5
            max_epoch = epoch_list[i]
            
            ts_result[1] = [1, precision_list[i], recall_list[i], acc_list[i], 
                           f1_list[i], auc_list[i], sen_list[i], spe_list[i]]
            ts_result[2] = [2, precision_list[i + num], recall_list[i + num], 
                           acc_list[i + num], f1_list[i + num], auc_list[i + num], 
                           sen_list[i + num], spe_list[i + num]]
            ts_result[3] = [3, precision_list[i + 2 * num], recall_list[i + 2 * num], 
                           acc_list[i + 2 * num], f1_list[i + 2 * num], auc_list[i + 2 * num], 
                           sen_list[i + 2 * num], spe_list[i + 2 * num]]
            ts_result[4] = [4, precision_list[i + 3 * num], recall_list[i + 3 * num], 
                           acc_list[i + 3 * num], f1_list[i + 3 * num], auc_list[i + 3 * num], 
                           sen_list[i + 3 * num], spe_list[i + 3 * num]]
            ts_result[5] = [5, precision_list[i + 4 * num], recall_list[i + 4 * num], 
                           acc_list[i + 4 * num], f1_list[i + 4 * num], auc_list[i + 4 * num], 
                           sen_list[i + 4 * num], spe_list[i + 4 * num]]
    
    print('{}-{}-{}-{}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}\n'.format(
        max_epoch, lr, args.batch_size, weight_decay, max_acc, max_precision,
        max_recall, max_f1, max_auc))
    
    # Save results
    save_xlsx(ts_result, args.model, args.site, max_epoch, lr, args.batch_size, weight_decay)
    
    if (ts_result[1][5] == 0.5 or ts_result[2][5] == 0.5 or ts_result[3][5] == 0.5 or 
        ts_result[4][5] == 0.5 or ts_result[5][5] == 0.5):
        print("Model not fitting properly: Test AUC == 0.5")
        return 0
    
    return max_acc * 100
