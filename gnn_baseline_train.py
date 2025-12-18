"""
Training script for GAT Baseline Model

This module implements the training and evaluation pipeline for the GAT baseline model,
including 5-fold cross-validation, early stopping, learning rate scheduling, and
comprehensive loss functions (Focal Loss, TopK Loss, Diversity Loss, L1 regularization).
"""
from utils import save_model_with_checks, setup_seed, save_xlsx, count_parameters, load_demographic_data
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
from prompt_utils import generate_subject_prompt, generate_disease_prompts, get_class_names_for_dataset
import warnings
warnings.filterwarnings("ignore")


def make_model_baseline_model(num_nodes=116, input_dim=116, hidden_dim=128, num_heads=2,
                              num_layers=2, dropout=0.5, num_classes=2, temporal_strategy='mean', 
                              device=0, use_input_norm=False, use_layernorm_gat=False, use_residual=False,
                              fc_hidden_dim=64, fc_num_layers=1, fc_activation='relu', fc_use_batchnorm=False,
                              ablation_mode='full', ablation_mask='with_mask', ablation_pooling='attention',
                              use_subject_prompt=False, use_disease_prompt=False, text_encoder=None):
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
        use_input_norm (bool): Enable BatchNorm1d after prompt. Default: False
        use_layernorm_gat (bool): Enable LayerNorm between GAT layers. Default: False
        use_residual (bool): Enable residual connections. Default: False
    
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
        temporal_strategy=temporal_strategy,
        use_input_norm=use_input_norm,
        use_layernorm_gat=use_layernorm_gat,
        use_residual=use_residual,
        fc_hidden_dim=fc_hidden_dim,
        fc_num_layers=fc_num_layers,
        fc_activation=fc_activation,
        fc_use_batchnorm=fc_use_batchnorm,
        ablation_mode=ablation_mode,
        ablation_mask=ablation_mask,
        ablation_pooling=ablation_pooling,
        use_subject_prompt=use_subject_prompt,
        use_disease_prompt=use_disease_prompt,
        text_encoder=text_encoder
    )
    
    # Initialize parameters (optimized initialization)
    # NOTE: fc1.weight and fc1.bias are already initialized in model.__init__
    # with optimized values, so we skip them to avoid overwriting
    for name, p in model.named_parameters():
        if name == "soft_prompt":
            continue  # soft_prompt is initialized in __init__ based on ablation_mode
        if name == "fc1.weight" or name == "fc1.bias" or name == "fc_final.weight" or name == "fc_final.bias":
            continue  # Skip final layers - already initialized in __init__ with optimized values
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
        
        # Use weight decay from arguments (default 0.0, but can be set for regularization)
        # Previously forced to 0.0, now allows user to set for anti-overfitting
        if decay is None or decay < 0:
            decay = 0.0
        print(f"[Weight Decay] Using value: {decay}")
        
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
    
    # Load demographic data if subject prompts are enabled
    demographics = {}
    index_to_file_id = []
    use_subject_prompt = getattr(args, 'use_subject_prompt', False)
    use_disease_prompt = getattr(args, 'use_disease_prompt', False)
    
    if use_subject_prompt or use_disease_prompt:
        demographics, index_to_file_id = load_demographic_data(args.site)
        if not demographics:
            print(f"Warning: No demographic data found for site {args.site}")
            if use_subject_prompt:
                print("  Disabling subject prompts")
                use_subject_prompt = False
            if use_disease_prompt:
                print("  Disabling disease prompts")
                use_disease_prompt = False
    
    # Initialize text encoder if needed (shared Llama-encoder-1.0B)
    text_encoder = None
    if use_subject_prompt or use_disease_prompt:
        try:
            from llm2vec import LLM2Vec
            text_encoder = LLM2Vec.from_pretrained("knowledgator/Llama-encoder-1.0B")
            text_encoder.eval()  # Freeze encoder
            for param in text_encoder.parameters():
                param.requires_grad = False
            print("✓ Loaded Llama-encoder-1.0B for multi-level prompts")
        except ImportError:
            try:
                from transformers import AutoModel, AutoTokenizer
                class TextEncoderWrapper:
                    def __init__(self):
                        self.tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
                        self.model = AutoModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
                        self.model.eval()
                        for param in self.model.parameters():
                            param.requires_grad = False
                text_encoder = TextEncoderWrapper()
                print("✓ Loaded Llama-encoder-1.0B via transformers")
            except Exception as e:
                print(f"Warning: Could not load text encoder: {e}")
                print("  Disabling subject and disease prompts")
                use_subject_prompt = False
                use_disease_prompt = False
    
    # Generate disease prompts if enabled
    disease_prompt_embeddings = None
    if use_disease_prompt and text_encoder is not None:
        class_names = get_class_names_for_dataset('ABIDE', num_classes=2)
        disease_prompts = generate_disease_prompts(class_names, 'ABIDE')
        
        # Encode disease prompts
        with torch.no_grad():
            if hasattr(text_encoder, 'encode'):
                # LLM2Vec interface
                disease_embeddings = text_encoder.encode(disease_prompts, convert_to_numpy=False, show_progress_bar=False)
                if isinstance(disease_embeddings, torch.Tensor):
                    disease_prompt_embeddings = disease_embeddings.to(device)
                else:
                    disease_prompt_embeddings = torch.tensor(disease_embeddings, device=device)
            else:
                # Transformers interface
                disease_emb_list = []
                for prompt in disease_prompts:
                    inputs = text_encoder.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = text_encoder.model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze()
                    disease_emb_list.append(emb)
                disease_prompt_embeddings = torch.stack(disease_emb_list).to(device)
        
        print(f"✓ Encoded {len(disease_prompts)} disease prompts")
    
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
        
        # Store train_index and test_index for demographic mapping
        # These are global indices that map to index_to_file_id
        
        print('X_train{}'.format(X_train.shape))
        print('X_test{}'.format(X_test.shape))
        print('Y_train{}'.format(Y_train.shape))
        print('Y_test{}'.format(Y_test.shape))
        
        # Create model
        # Disable improvements by default to restore original behavior
        # Can be enabled individually to test impact
        use_input_norm = getattr(args, 'use_input_norm', False)
        use_layernorm_gat = getattr(args, 'use_layernorm_gat', False)  # Disabled by default
        use_residual = getattr(args, 'use_residual', False)  # Disabled by default
        
        model = make_model_baseline_model(
            num_nodes=116,
            input_dim=116,  # Each node gets its row from correlation matrix
            hidden_dim=args.gat_hidden_dim,
            num_heads=args.gat_heads,
            num_layers=args.gat_num_layers,
            dropout=args.dropout,
            num_classes=2,
            temporal_strategy=args.temporal_strategy,
            device=device,
            use_input_norm=use_input_norm,
            use_layernorm_gat=use_layernorm_gat,
            use_residual=use_residual,
            fc_hidden_dim=getattr(args, 'fc_hidden_dim', 64),
            fc_num_layers=getattr(args, 'fc_num_layers', 1),
            fc_activation=getattr(args, 'fc_activation', 'relu'),
            fc_use_batchnorm=getattr(args, 'fc_use_batchnorm', False),
            ablation_mode=getattr(args, 'ablation_mode', 'full'),
            ablation_mask=getattr(args, 'ablation_mask', 'with_mask'),
            ablation_pooling=getattr(args, 'ablation_pooling', 'attention'),
            use_subject_prompt=use_subject_prompt,
            use_disease_prompt=use_disease_prompt,
            text_encoder=text_encoder
        )
        
        # Set disease prompt embeddings in model if available
        if use_disease_prompt and disease_prompt_embeddings is not None:
            # Project to match hidden_dim if needed
            if disease_prompt_embeddings.shape[1] != args.gat_hidden_dim:
                disease_proj = nn.Linear(disease_prompt_embeddings.shape[1], args.gat_hidden_dim).to(device)
                nn.init.xavier_uniform_(disease_proj.weight)
                with torch.no_grad():
                    disease_prompt_embeddings = disease_proj(disease_prompt_embeddings)
            model.disease_prompt_embeddings = disease_prompt_embeddings
        
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
        val_indices_local = np.random.choice(len(X_train), size=val_size, replace=False)
        train_indices_local = np.setdiff1d(np.arange(len(X_train)), val_indices_local)
        
        X_train_final = X_train[train_indices_local]
        X_val = X_train[val_indices_local]
        X_mask_train_final = X_mask_train[train_indices_local]
        X_mask_val = X_mask_train[val_indices_local]
        Y_train_final = Y_train[train_indices_local]
        Y_val = Y_train[val_indices_local]
        
        # Class weights (based on training set of this fold)
        num_class_0 = (Y_train_final == 0).sum()
        num_class_1 = (Y_train_final == 1).sum()
        total_samples = len(Y_train_final)
        weight_0 = total_samples / (2 * num_class_0 + 1e-6)
        weight_1 = total_samples / (2 * num_class_1 + 1e-6)
        
        # Choose loss function based on args
        loss_type = getattr(args, 'loss_type', 'focal')
        if loss_type == 'bce':
            # Binary Cross Entropy with class weights
            class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            print(f'  Using BCE Loss (CrossEntropyLoss) with class weights: [{weight_0:.4f}, {weight_1:.4f}]')
        else:
            # Focal Loss
            focal_alpha = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
            focal_gamma = getattr(args, 'focal_gamma', 1.0)
            loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction='mean', num_classes=2)
            print(f'  Using Focal Loss (gamma={focal_gamma}) with alpha: [{weight_0:.4f}, {weight_1:.4f}]')
        
        # Log class balance
        print(f'\n[CLASS BALANCE] Fold {kfold_index}:')
        print(f'  Train: Class 0={num_class_0} ({num_class_0/total_samples*100:.1f}%), Class 1={num_class_1} ({num_class_1/total_samples*100:.1f}%)')
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
                    # Use configurable threshold (default 0.0 = dense graph)
                    threshold=args.edge_threshold if args.edge_threshold > 0.0 else None,
                    temporal_strategy=args.temporal_strategy,
                    window_mask=train_mask_batch
                )
                batch_data = batch_data.to(device)
                batch_data.y = train_label_batch.to(device)
                
                # Generate subject prompts for this batch if enabled
                subject_prompts_batch = None
                if use_subject_prompt and demographics and index_to_file_id:
                    subject_prompts_batch = []
                    total_sites = 16  # ABIDE has 16 sites
                    for batch_idx in idx_batch:
                        # batch_idx is local to X_train_final, map to global index
                        global_idx = train_index[train_indices_local[batch_idx]]
                        if global_idx < len(index_to_file_id):
                            file_id = index_to_file_id[global_idx]
                            demo = demographics.get(file_id, {})
                            age = demo.get('age', 25.0) if demo.get('age') is not None else 25.0
                            sex = demo.get('sex', 1) if demo.get('sex') is not None else 1
                            site_id = demo.get('site_id', args.site) if demo.get('site_id') else args.site
                        else:
                            # Fallback if index out of range
                            age = 25.0
                            sex = 1
                            site_id = args.site
                        prompt = generate_subject_prompt(age, sex, site_id, total_sites)
                        subject_prompts_batch.append(prompt)
                
                outputs, attention_weights, features_intermediate = model(
                    batch_data, 
                    subject_prompts=subject_prompts_batch,
                    demographics=demographics if use_subject_prompt else None
                )
                
                # Use selected loss function
                loss = loss_fn(outputs, batch_data.y)
                
                # Disease-level prompt loss (L_disease)
                loss_disease = torch.tensor(0.0, device=device, requires_grad=True)
                if use_disease_prompt and disease_prompt_embeddings is not None:
                    # features_intermediate is the representation after Population Graph (if enabled)
                    # Shape: (batch_size, hidden_dim)
                    subject_reps = features_intermediate  # (batch_size, hidden_dim)
                    
                    # Normalize for cosine similarity
                    subject_reps_norm = F.normalize(subject_reps, p=2, dim=1)
                    disease_emb_norm = F.normalize(disease_prompt_embeddings, p=2, dim=1)
                    
                    # Compute similarity matrix: (batch_size, num_classes)
                    similarity_matrix = torch.mm(subject_reps_norm, disease_emb_norm.T)
                    
                    # Use cross-entropy: treat similarity as logits
                    # The true class should have highest similarity
                    loss_disease = F.cross_entropy(similarity_matrix, batch_data.y)
                    
                    # Add to total loss with weight lambda_disease
                    lambda_disease = getattr(args, 'lambda_disease', 1.0)
                    loss = loss + lambda_disease * loss_disease
                
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
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
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
                            threshold=args.edge_threshold if args.edge_threshold > 0.0 else None,
                            temporal_strategy=args.temporal_strategy,
                            window_mask=train_mask_batch
                        )
                        batch_data = batch_data.to(device)
                        
                        # Generate subject prompts for train batch if enabled
                        train_subject_prompts = None
                        if use_subject_prompt and demographics and index_to_file_id:
                            train_subject_prompts = []
                            total_sites = 16
                            for batch_idx in idx_batch:
                                # batch_idx is local to X_train_final, map to global index
                                global_idx = train_index[train_indices_local[batch_idx]]
                                if global_idx < len(index_to_file_id):
                                    file_id = index_to_file_id[global_idx]
                                    demo = demographics.get(file_id, {})
                                    age = demo.get('age', 25.0) if demo.get('age') is not None else 25.0
                                    sex = demo.get('sex', 1) if demo.get('sex') is not None else 1
                                    site_id = demo.get('site_id', args.site) if demo.get('site_id') else args.site
                                else:
                                    age = 25.0
                                    sex = 1
                                    site_id = args.site
                                prompt = generate_subject_prompt(age, sex, site_id, total_sites)
                                train_subject_prompts.append(prompt)
                        
                        outputs, _, _ = model(
                            batch_data,
                            subject_prompts=train_subject_prompts,
                            demographics=demographics if use_subject_prompt else None
                        )
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
                            threshold=args.edge_threshold if args.edge_threshold > 0.0 else None,
                            temporal_strategy=args.temporal_strategy,
                            window_mask=val_mask_batch
                        )
                        batch_data_val = batch_data_val.to(device)
                        batch_data_val.y = torch.from_numpy(batch_val_Y).long().to(device)
                        
                        # Generate subject prompts for validation batch if enabled
                        val_subject_prompts = None
                        if use_subject_prompt and demographics and index_to_file_id:
                            val_subject_prompts = []
                            total_sites = 16
                            val_batch_indices = val_indices_local[start_idx:end_idx]
                            for val_batch_idx in val_batch_indices:
                                # Map to global index: val_batch_idx is local to X_train, map to original dataset
                                global_idx = train_index[val_batch_idx]
                                if global_idx < len(index_to_file_id):
                                    file_id = index_to_file_id[global_idx]
                                    demo = demographics.get(file_id, {})
                                    age = demo.get('age', 25.0) if demo.get('age') is not None else 25.0
                                    sex = demo.get('sex', 1) if demo.get('sex') is not None else 1
                                    site_id = demo.get('site_id', args.site) if demo.get('site_id') else args.site
                                else:
                                    age = 25.0
                                    sex = 1
                                    site_id = args.site
                                prompt = generate_subject_prompt(age, sex, site_id, total_sites)
                                val_subject_prompts.append(prompt)
                        
                        outputs_val, _, _ = model(
                            batch_data_val,
                            subject_prompts=val_subject_prompts,
                            demographics=demographics if use_subject_prompt else None
                        )
                        val_loss = loss_fn(outputs_val, batch_data_val.y)
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
                    threshold=args.edge_threshold if args.edge_threshold > 0.0 else None,
                    temporal_strategy=args.temporal_strategy,
                    window_mask=test_mask_batch
                )
                batch_data = batch_data.to(device)
                
                # Generate subject prompts for test set if enabled
                test_subject_prompts = None
                if use_subject_prompt and demographics and index_to_file_id:
                    test_subject_prompts = []
                    total_sites = 16
                    for test_local_idx in range(len(X_test)):
                        # Map to global index
                        global_test_idx = test_index[test_local_idx]
                        if global_test_idx < len(index_to_file_id):
                            file_id = index_to_file_id[global_test_idx]
                            demo = demographics.get(file_id, {})
                            age = demo.get('age', 25.0) if demo.get('age') is not None else 25.0
                            sex = demo.get('sex', 1) if demo.get('sex') is not None else 1
                            site_id = demo.get('site_id', args.site) if demo.get('site_id') else args.site
                        else:
                            age = 25.0
                            sex = 1
                            site_id = args.site
                        prompt = generate_subject_prompt(age, sex, site_id, total_sites)
                        test_subject_prompts.append(prompt)
                
                outputs, _, features_intermediate = model(
                    batch_data,
                    subject_prompts=test_subject_prompts,
                    demographics=demographics if use_subject_prompt else None
                )
                
                # Clear memory after inference
                del batch_data
                
                # Use softmax to get probabilities and then argmax
                probs = F.softmax(outputs, dim=1)
                _, indices = torch.max(probs, dim=1)
                pre = indices.cpu().numpy()
                label_test = Y_test
                
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
