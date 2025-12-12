"""
GAT Baseline Model for BrainPromptGAT

This module implements a Graph Attention Network (GAT) baseline model for brain network analysis,
adapting the original BrainPrompt Transformer architecture to work with graph-structured data.

The model processes functional connectivity matrices (116x116) as graphs where:
- Nodes: 116 ROIs (Regions of Interest)
- Edges: Functional connections (correlations between ROIs)
- Edge weights: Correlation values

Key components:
1. Semantic prompt injection (LLM-based BERT embeddings)
2. Learnable global adjacency mask (consensus graph)
3. Multi-layer GAT encoder
4. Global attention pooling
5. Classification layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.utils import softmax, scatter
import os
import warnings
warnings.filterwarnings("ignore")


class GATBaselineModel(nn.Module):
    """
    Graph Attention Network Baseline Model for Brain Network Classification
    
    This model replaces the Transformer encoder from the original BrainPrompt with a GAT-based
    architecture that explicitly models the graph structure of functional connectivity networks.
    
    Args:
        num_nodes (int): Number of nodes (ROIs) in the graph. Default: 116 (AAL atlas)
        input_dim (int): Input feature dimension for each node. Default: 116 (full row of correlation matrix)
        hidden_dim (int): Hidden dimension for GAT layers. Default: 128
        num_heads (int): Number of attention heads in first GAT layer. Default: 2
        num_layers (int): Number of GAT layers. Default: 2
        dropout (float): Dropout rate. Default: 0.5
        num_classes (int): Number of output classes. Default: 2 (binary classification)
        temporal_strategy (str): Strategy for temporal aggregation. Default: 'mean'
    
    Architecture:
        1. Semantic prompt injection (BERT embeddings + trainable soft prompt)
        2. Learnable global adjacency mask (filters noisy connections)
        3. Multi-layer GAT encoder with edge attributes
        4. Global attention pooling (aggregates node features to graph-level)
        5. Classification layers (FC + ReLU + Dropout + FC)
    """
    
    def __init__(self, num_nodes=116, input_dim=116, hidden_dim=128, num_heads=2,
                 num_layers=2, dropout=0.5, num_classes=2, temporal_strategy='mean'):
        super(GATBaselineModel, self).__init__()
        
        # --- Semantic Prompt (LLM-based) ---
        # Loads pre-trained BERT embeddings for each ROI to provide semantic initialization
        # The prompt consists of:
        # - static_prompt: Pre-trained BERT embeddings (non-trainable, 116x384)
        # - prompt_projector: Projects BERT dim (384) to input_dim (116)
        # - soft_prompt: Trainable delta that fine-tunes the semantic prompt (116x116)
        self.static_prompt = None  # Fixed textual embedding (non-trainable)
        self.prompt_projector = None  # Projects dim_text -> input_dim
        self.soft_prompt = nn.Parameter(torch.zeros(num_nodes, input_dim))  # Trainable delta
        
        emb_path = os.path.join(os.path.dirname(__file__), "data", "roi_bert_embeddings.pt")
        try:
            pretrained = torch.load(emb_path, map_location="cpu")
            if pretrained.dim() == 2 and pretrained.shape[0] == num_nodes:
                self.static_prompt = pretrained
                self.prompt_projector = nn.Linear(pretrained.shape[1], input_dim)
                nn.init.xavier_uniform_(self.prompt_projector.weight)
                nn.init.zeros_(self.prompt_projector.bias)
                # More generous initialization for soft prompt to allow effective learning
                # std=0.1 provides sufficient variation without destabilizing (10x larger than before)
                nn.init.normal_(self.soft_prompt, std=0.1)
                print(f"✓ Loaded semantic prompt from {emb_path} (shape={pretrained.shape})")
            else:
                print(f"! Prompt at {emb_path} has unexpected shape {getattr(pretrained, 'shape', None)}, ignoring.")
        except FileNotFoundError:
            print(f"! Embeddings file not found at {emb_path}; using random trainable prompt.")
            nn.init.xavier_uniform_(self.soft_prompt)
        # -----------------------------------

        self.num_nodes = num_nodes
        self.temporal_strategy = temporal_strategy
        
        # Input normalization (stabilizes prompt + image sum)
        # NOTE: LayerNorm can zero features if std is too small
        # Disabled to avoid zeroing features when variance is very low
        self.input_norm = None  # Disable normalization after prompt to avoid zeroing features
        
        # GAT Encoder
        # First layer: input_dim -> hidden_dim with multiple heads
        self.conv1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True, edge_dim=1,
            add_self_loops=False  # Self-connections already in correlation matrix
        )
        
        # Intermediate GAT layers
        # After first layer: hidden_dim*num_heads -> hidden_dim (single head)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                # Second layer: hidden_dim*num_heads -> hidden_dim
                self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, 
                                         dropout=dropout, concat=False, edge_dim=1,
                                         add_self_loops=False))
            else:
                # Additional layers: hidden_dim -> hidden_dim
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, 
                                         dropout=dropout, concat=False, edge_dim=1,
                                         add_self_loops=False))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension after pooling
        # Using GlobalAttention Pooling (returns hidden_dim)
        pool_dim = hidden_dim  
        
        # Global Attention Pooling Layer
        # Gate network maps node features to a scalar attention score
        self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        
        # Learnable Global Adjacency Mask (Consensus Graph)
        # Inspired by BrainPrompt Transformer (target_model.py)
        # The model learns which connections are globally relevant, filtering noisy edges
        # This is a 116x116 learnable parameter that acts as a global filter
        self.global_adj = nn.Parameter(torch.Tensor(116, 116))
        nn.init.uniform_(self.global_adj, a=0, b=1)  # Initialize uniformly like reference
        
        # Use LayerNorm instead of BatchNorm1d since global_adj is a single parameter (116, 116)
        # not a batch. LayerNorm normalizes over features (last dimension)
        self.ln_global_adj = nn.LayerNorm(116)
        
        # Classification layers
        # NOTE: LayerNorm removed as it was zeroing features when std was too small
        # LayerNorm can zero features if variance is very low
        # Components separated to allow detailed logging if needed
        self.fc_linear = nn.Linear(pool_dim, 64)
        # self.fc_layernorm = nn.LayerNorm(64)  # Removed - was zeroing features
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, num_classes)
        
        # More aggressive initialization for final layers (generate larger outputs)
        # gain=2.0 increases output scale to avoid very small logits
        # Non-zero bias helps break initial symmetry
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=0.1)  # Small but non-zero bias
    
    def get_mask_prompt_l1_norm(self):
        """
        Returns the L1 norm of the soft prompt for regularization.
        Useful for adding L1 regularization to the loss during training.
        
        Returns:
            torch.Tensor: L1 norm normalized by number of elements
        """
        return torch.norm(self.soft_prompt, p=1) / self.soft_prompt.numel()
    
    def forward(self, batch_data, debug=False):
        """
        Forward pass through the GAT model.
        
        Args:
            batch_data: PyTorch Geometric Batch object containing:
                - x: Node features [Batch_Size * Num_Nodes, Input_Dim]
                - edge_index: Edge connectivity [2, Num_Edges]
                - edge_attr or edge_weight: Edge attributes [Num_Edges] or [Num_Edges, 1]
                - batch: Batch assignment for each node [Batch_Size * Num_Nodes]
            debug (bool): If True, prints detailed debug information. Default: False
        
        Returns:
            tuple: (outputs, attention_weights, intermediate_features)
                - outputs: Classification logits [Batch_Size, num_classes]
                - attention_weights: Node attention weights from pooling [Num_Nodes, 1] or None
                - intermediate_features: Features before final classification layer [Batch_Size, 64]
        """
        x = batch_data.x  # Shape: [Batch_Size * Num_Nodes, Input_Dim]
        edge_index = batch_data.edge_index
        edge_attr = getattr(batch_data, "edge_weight", None)
        if edge_attr is None:
            edge_attr = getattr(batch_data, "edge_attr", None)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # Compatible with edge_dim=1
        batch = batch_data.batch
        
        # Debug logging: input statistics
        if debug:
            print(f"\n[FORWARD DEBUG] Input x: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
            if edge_attr is not None:
                print(f"[FORWARD DEBUG] Input edge_attr: mean={edge_attr.mean():.4f}, std={edge_attr.std():.4f}, "
                      f"min={edge_attr.min():.4f}, max={edge_attr.max():.4f}, shape={edge_attr.shape}")
            else:
                print(f"[FORWARD DEBUG] Input edge_attr: None")
            print(f"[FORWARD DEBUG] batch: shape={batch.shape if batch is not None else None}, "
                  f"unique_batches={torch.unique(batch).tolist() if batch is not None else None}")

        # --- Prompt Injection (Additive with semantic initialization) ---
        # Injects semantic information from BERT embeddings into node features
        # The prompt is added to the input features: x = x + prompt
        if x.shape[0] % self.num_nodes == 0:
            num_graphs = x.shape[0] // self.num_nodes
            
            # Combine static (BERT) and soft (trainable) prompts
            if self.static_prompt is not None and self.prompt_projector is not None:
                projected = self.prompt_projector(self.static_prompt.to(x.device))
                base_prompt = projected + self.soft_prompt
            else:
                base_prompt = self.soft_prompt
            
            # Repeat prompt for each graph in the batch
            prompt_batch = base_prompt.repeat(num_graphs, 1)
            x = x + prompt_batch
            
            # Apply normalization only if enabled
            # LayerNorm was zeroing features when std was too small
            if self.input_norm is not None:
                x = self.input_norm(x)
            
            if debug:
                print(f"[FORWARD DEBUG] After prompt injection: mean={x.mean():.4f}, std={x.std():.4f}, "
                      f"min={x.min():.4f}, max={x.max():.4f}, norm_applied={self.input_norm is not None}")
        else:
            # Fallback (should not occur with fixed atlas)
            if debug:
                print(f"[FORWARD DEBUG] Warning: x.shape[0] ({x.shape[0]}) not divisible by num_nodes ({self.num_nodes})")
        
        # --- Learnable Global Adjacency (Consensus Graph) ---
        # Inspired by BrainPrompt Transformer (target_model.py)
        # The model learns which connections are globally relevant, filtering noisy edges
        global_adj = self.global_adj  # (116, 116)
        
        # Normalization and Activation
        # LayerNorm normalizes each row (ROI) independently, which makes sense
        # for an adjacency matrix where each row represents connections of a ROI
        global_adj = F.relu(self.ln_global_adj(global_adj))
        
        # Symmetrize (Functional Connectivity is non-directional)
        global_adj = (global_adj + global_adj.T) / 2
        
        # Generate soft mask (0 to 1)
        adj_mask = torch.sigmoid(global_adj)
        
        # Apply mask to input graph edges
        # edge_index: [2, E] - edge connectivity
        # edge_attr: [E] or [E, 1] - edge weights
        row, col = edge_index
        
        # Find mask weight corresponding to each edge (u, v)
        # IMPORTANT: In PyG batches, node indices are accumulated (0..115, 116..231, etc)
        # We need to map back to canonical ROI index (0..115) using modulo
        num_nodes_roi = 116
        row_mod = row % num_nodes_roi
        col_mod = col % num_nodes_roi
        
        # mask_values will have shape [E]
        mask_values = adj_mask[row_mod, col_mod]
        
        # Re-weight edges: Final_Weight = Original_Weight * Global_Mask
        # This filters noisy edges that the model has learned are useless
        edge_attr_weighted = None
        if edge_attr is not None:
             # If edge_attr is 2D (e.g., [E, 1]), expand mask_values to match
             if edge_attr.dim() == 2 and edge_attr.shape[1] == 1:
                 edge_attr_weighted = edge_attr * mask_values.unsqueeze(-1)
             else:  # Assume edge_attr is 1D or other shape where direct multiplication is intended
                 edge_attr_weighted = edge_attr * mask_values
             if debug:
                 print(f"[FORWARD DEBUG] Global adj mask: mean={adj_mask.mean():.4f}, std={adj_mask.std():.4f}, "
                       f"min={adj_mask.min():.4f}, max={adj_mask.max():.4f}")
                 print(f"[FORWARD DEBUG] mask_values: mean={mask_values.mean():.4f}, std={mask_values.std():.4f}, "
                       f"min={mask_values.min():.4f}, max={mask_values.max():.4f}")
                 print(f"[FORWARD DEBUG] edge_attr_weighted: mean={edge_attr_weighted.mean():.4f}, "
                       f"std={edge_attr_weighted.std():.4f}, min={edge_attr_weighted.min():.4f}, "
                       f"max={edge_attr_weighted.max():.4f}, shape={edge_attr_weighted.shape}")
        elif debug:
            print(f"[FORWARD DEBUG] edge_attr is None, edge_attr_weighted will be None")
        
        # GAT Layers
        # Pass edge_attr_weighted to all layers for consistency
        # The learned global mask should be applied in all layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr_weighted)
        if debug:
            print(f"[FORWARD DEBUG] After conv1 (before ELU): mean={x.mean():.4f}, std={x.std():.4f}, "
                  f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
        x = F.elu(x)
        if debug:
            print(f"[FORWARD DEBUG] After conv1 + ELU: mean={x.mean():.4f}, std={x.std():.4f}, "
                  f"min={x.min():.4f}, max={x.max():.4f}")
        x = self.dropout(x)
        if debug:
            print(f"[FORWARD DEBUG] After conv1 + dropout: mean={x.mean():.4f}, std={x.std():.4f}, "
                  f"min={x.min():.4f}, max={x.max():.4f}, dropout_rate={self.dropout.p}, "
                  f"training_mode={self.training}, zero_ratio={(x == 0).float().mean():.4f}")
        
        # Additional GAT layers - use edge_attr_weighted in all layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr_weighted)
            if debug:
                print(f"[FORWARD DEBUG] After conv{i+2} (before ELU): mean={x.mean():.4f}, std={x.std():.4f}, "
                      f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
            x = F.elu(x)
            if debug:
                print(f"[FORWARD DEBUG] After conv{i+2} + ELU: mean={x.mean():.4f}, std={x.std():.4f}, "
                      f"min={x.min():.4f}, max={x.max():.4f}")
            x = self.dropout(x)
            if debug:
                print(f"[FORWARD DEBUG] After conv{i+2} + dropout: mean={x.mean():.4f}, std={x.std():.4f}, "
                      f"min={x.min():.4f}, max={x.max():.4f}, training_mode={self.training}, "
                      f"zero_ratio={(x == 0).float().mean():.4f}")
        
        # Global pooling if batch is provided
        # Global Attention Pooling (GAT-LI method)
        # Manually implemented to expose attention weights for TopK Loss
        weights = None
        if batch is not None:
             if debug:
                 print(f"[FORWARD DEBUG] Before pooling: x mean={x.mean():.4f}, std={x.std():.4f}, "
                       f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
             
             # 1. Get scores for each node: gate_nn(x) -> (N, 1)
             scores = self.pool.gate_nn(x).view(-1, 1)  # Ensure (N, 1)
             if debug:
                 print(f"[FORWARD DEBUG] Pooling scores: mean={scores.mean():.4f}, std={scores.std():.4f}, "
                       f"min={scores.min():.4f}, max={scores.max():.4f}, shape={scores.shape}")
             
             # 2. Calculate weights via Softmax (per graph in batch)
             weights = softmax(scores, batch)  # (N, 1)
             if debug:
                 weights_sum_per_graph = scatter(weights, batch, dim=0, reduce='sum')
                 print(f"[FORWARD DEBUG] Pooling weights (after softmax): mean={weights.mean():.4f}, "
                       f"std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}, "
                       f"shape={weights.shape}, sum_per_graph={weights_sum_per_graph.tolist()}")
             
             # 3. Weighted Sum: x_pool = sum(weights * x)
             # Aggregates weighted features per graph
             x_weighted = x * weights
             if debug:
                 print(f"[FORWARD DEBUG] x * weights: mean={x_weighted.mean():.4f}, std={x_weighted.std():.4f}, "
                       f"min={x_weighted.min():.4f}, max={x_weighted.max():.4f}, shape={x_weighted.shape}")
             x = scatter(x_weighted, batch, dim=0, reduce='add')  # (Batch, Hidden)
             if debug:
                 print(f"[FORWARD DEBUG] After pooling (scatter): mean={x.mean():.4f}, std={x.std():.4f}, "
                       f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
             
        else:
             # Fallback: mean pooling if no batch information
             if debug:
                 print(f"[FORWARD DEBUG] batch is None, using mean pooling")
             x = x.mean(dim=0, keepdim=True)
             weights = torch.ones(x.shape[0], 1).to(x.device) / x.shape[0]
             if debug:
                 print(f"[FORWARD DEBUG] After mean pooling: mean={x.mean():.4f}, std={x.std():.4f}, "
                       f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")

        # Classification layers
        # Store intermediate features for debugging (before final layer)
        if debug:
            print(f"[FORWARD DEBUG] Before fc: mean={x.mean():.4f}, std={x.std():.4f}, "
                  f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
        
        # Process through fc with detailed logging
        x_after_linear = self.fc_linear(x)
        if debug:
            print(f"[FORWARD DEBUG] After fc Linear: mean={x_after_linear.mean():.4f}, "
                  f"std={x_after_linear.std():.4f}, min={x_after_linear.min():.4f}, "
                  f"max={x_after_linear.max():.4f}, shape={x_after_linear.shape}")
        
        # LayerNorm removed - was zeroing features
        # x_after_layernorm = self.fc_layernorm(x_after_linear)
        
        x_after_relu = self.fc_relu(x_after_linear)
        if debug:
            print(f"[FORWARD DEBUG] After fc ReLU: mean={x_after_relu.mean():.4f}, "
                  f"std={x_after_relu.std():.4f}, min={x_after_relu.min():.4f}, "
                  f"max={x_after_relu.max():.4f}, zero_ratio={(x_after_relu == 0).float().mean():.4f}")
        
        x_intermediate = self.fc_dropout(x_after_relu)
        if debug:
            print(f"[FORWARD DEBUG] After fc Dropout: mean={x_intermediate.mean():.4f}, "
                  f"std={x_intermediate.std():.4f}, min={x_intermediate.min():.4f}, "
                  f"max={x_intermediate.max():.4f}, training_mode={self.training}, "
                  f"zero_ratio={(x_intermediate == 0).float().mean():.4f}, shape={x_intermediate.shape}")
        
        x = self.fc1(x_intermediate)  # (Batch, num_classes)
        if debug:
            print(f"[FORWARD DEBUG] After fc1 (outputs): mean={x.mean():.4f}, std={x.std():.4f}, "
                  f"min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}\n")
        
        # Return prediction, attention weights (for TopK Loss), and intermediate features (for debugging)
        return x, weights, x_intermediate
