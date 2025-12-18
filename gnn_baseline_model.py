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
        use_input_norm (bool): Enable BatchNorm1d after prompt injection. Default: False
        use_layernorm_gat (bool): Enable LayerNorm between GAT layers. Default: False
        use_residual (bool): Enable residual connections between GAT layers. Default: False
        fc_hidden_dim (int): Hidden dimension for FC classification layers. Default: 64
        fc_num_layers (int): Number of hidden FC layers (1, 2, or 3). Default: 1
        fc_activation (str): Activation function for FC layers ('relu', 'elu', 'gelu'). Default: 'relu'
        fc_use_batchnorm (bool): Use BatchNorm in FC layers. Default: False
    
    Architecture:
        1. Semantic prompt injection (BERT embeddings + trainable soft prompt)
        2. Learnable global adjacency mask (filters noisy connections)
        3. Multi-layer GAT encoder with edge attributes (uses ELU activation)
        4. Global attention pooling (aggregates node features to graph-level)
        5. Classification layers (configurable: FC layers + activation + optional BatchNorm + Dropout + FC)
    
    Note on ELU vs ReLU:
        - GAT layers use ELU: Produces activations with mean near zero, smooth negative gradients,
          helps with training stability in graph neural networks
        - FC layers use configurable activation (default ReLU): Can be changed to ELU or GELU
          for potentially better performance
    """
    
    def __init__(self, num_nodes=116, input_dim=116, hidden_dim=128, num_heads=2,
                 num_layers=2, dropout=0.5, num_classes=2, temporal_strategy='mean', 
                 use_input_norm=False, use_layernorm_gat=True, use_residual=True,
                 fc_hidden_dim=64, fc_num_layers=1, fc_activation='relu', fc_use_batchnorm=False,
                 ablation_mode='full', ablation_mask='with_mask', ablation_pooling='attention',
                 use_subject_prompt=False, use_disease_prompt=False, text_encoder=None):
        super(GATBaselineModel, self).__init__()
        
        # --- Semantic Prompt (LLM-based) ---
        # Ablation study: Different modes control prompt initialization and usage
        # - 'full': BERT embeddings + trainable soft prompt (default, best performance expected)
        # - 'no_prompt': No prompts at all (pure GAT baseline)
        # - 'random_prompt': Random initialized soft prompt (no BERT knowledge)
        # - 'static_prompt': BERT embeddings only (no fine-tuning via soft prompt)
        self.ablation_mode = ablation_mode
        self.static_prompt = None  # Fixed textual embedding (non-trainable)
        self.prompt_projector = None  # Projects dim_text -> input_dim
        self.soft_prompt = None  # Trainable delta (initialized based on ablation_mode)
        
        if ablation_mode == 'no_prompt':
            # No prompts: skip all prompt initialization
            print(f"[Ablation] Mode: no_prompt - No prompts will be used")
        elif ablation_mode == 'random_prompt':
            # Random prompt: initialize soft_prompt randomly, no BERT
            self.soft_prompt = nn.Parameter(torch.zeros(num_nodes, input_dim))
            nn.init.xavier_uniform_(self.soft_prompt)
            print(f"[Ablation] Mode: random_prompt - Using randomly initialized soft prompt (no BERT)")
        elif ablation_mode == 'static_prompt':
            # Static prompt: load BERT, but soft_prompt will be zeroed and non-trainable
            self.soft_prompt = nn.Parameter(torch.zeros(num_nodes, input_dim), requires_grad=False)
            emb_path = os.path.join(os.path.dirname(__file__), "data", "roi_bert_embeddings.pt")
            try:
                pretrained = torch.load(emb_path, map_location="cpu")
                if pretrained.dim() == 2 and pretrained.shape[0] == num_nodes:
                    self.static_prompt = pretrained
                    self.prompt_projector = nn.Linear(pretrained.shape[1], input_dim)
                    nn.init.xavier_uniform_(self.prompt_projector.weight)
                    nn.init.zeros_(self.prompt_projector.bias)
                    print(f"[Ablation] Mode: static_prompt - Using BERT embeddings only (no fine-tuning)")
                    print(f"✓ Loaded semantic prompt from {emb_path} (shape={pretrained.shape})")
                else:
                    print(f"! Prompt at {emb_path} has unexpected shape {getattr(pretrained, 'shape', None)}, ignoring.")
            except FileNotFoundError:
                print(f"! Embeddings file not found at {emb_path}; cannot use static_prompt mode.")
        else:  # ablation_mode == 'full'
            # Full model: BERT + trainable soft prompt (default behavior)
            self.soft_prompt = nn.Parameter(torch.zeros(num_nodes, input_dim))
            emb_path = os.path.join(os.path.dirname(__file__), "data", "roi_bert_embeddings.pt")
            try:
                pretrained = torch.load(emb_path, map_location="cpu")
                if pretrained.dim() == 2 and pretrained.shape[0] == num_nodes:
                    self.static_prompt = pretrained
                    self.prompt_projector = nn.Linear(pretrained.shape[1], input_dim)
                    nn.init.xavier_uniform_(self.prompt_projector.weight)
                    nn.init.zeros_(self.prompt_projector.bias)
                    # More generous initialization for soft prompt to allow effective learning
                    # std=0.1 provides sufficient variation without destabilizing
                    nn.init.normal_(self.soft_prompt, std=0.1)
                    print(f"[Ablation] Mode: full - Using BERT + trainable soft prompt")
                    print(f"✓ Loaded semantic prompt from {emb_path} (shape={pretrained.shape})")
                else:
                    print(f"! Prompt at {emb_path} has unexpected shape {getattr(pretrained, 'shape', None)}, ignoring.")
                    nn.init.xavier_uniform_(self.soft_prompt)
            except FileNotFoundError:
                print(f"! Embeddings file not found at {emb_path}; using random trainable prompt.")
                nn.init.xavier_uniform_(self.soft_prompt)
        # -----------------------------------

        self.num_nodes = num_nodes
        self.temporal_strategy = temporal_strategy
        
        # Input normalization (stabilizes prompt + image sum)
        # Disabled by default - was causing learning issues
        # Can be enabled by setting use_input_norm=True, but may need careful tuning
        self.input_norm = nn.BatchNorm1d(input_dim, track_running_stats=False) if use_input_norm else None
        
        # GAT Encoder
        # First layer: input_dim -> hidden_dim with multiple heads
        self.conv1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True, edge_dim=1,
            add_self_loops=False  # Self-connections already in correlation matrix
        )
        
        # Intermediate GAT layers
        # After first layer: hidden_dim*num_heads -> hidden_dim (single head)
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layernorm_gat else None  # LayerNorm for each intermediate GAT layer
        # Residual projection for first intermediate layer (hidden_dim*num_heads -> hidden_dim)
        self.residual_proj = None
        self.use_residual = use_residual
        self.use_layernorm_gat = use_layernorm_gat
        
        if num_layers > 1 and use_residual:
            self.residual_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        for i in range(num_layers - 1):
            if i == 0:
                # Second layer: hidden_dim*num_heads -> hidden_dim
                self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, 
                                         dropout=dropout, concat=False, edge_dim=1,
                                         add_self_loops=False))
                if use_layernorm_gat:
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))
            else:
                # Additional layers: hidden_dim -> hidden_dim
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, 
                                         dropout=dropout, concat=False, edge_dim=1,
                                         add_self_loops=False))
                if use_layernorm_gat:
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension after pooling
        # Using GlobalAttention Pooling (returns hidden_dim)
        pool_dim = hidden_dim  
        
        # Pooling Strategy (Ablation Study)
        # - 'attention': Global Attention Pooling (default, best performance expected)
        # - 'mean': Simple mean pooling
        # - 'max': Max pooling
        self.ablation_pooling = ablation_pooling
        if ablation_pooling == 'attention':
            # Global Attention Pooling Layer
            # Gate network maps node features to a scalar attention score
            # Create gate_nn separately to access it directly for manual pooling
            self.gate_nn: nn.Module = nn.Linear(hidden_dim, 1)
            self.pool = GlobalAttention(gate_nn=self.gate_nn)
            print(f"[Ablation] Pooling: attention (Global Attention Pooling)")
        else:
            # For mean/max pooling, we don't need gate_nn or pool object
            self.gate_nn = None
            self.pool = None
            print(f"[Ablation] Pooling: {ablation_pooling} (Simple {ablation_pooling.capitalize()} Pooling)")
        
        # Learnable Global Adjacency Mask (Consensus Graph) - Ablation Study
        # - 'with_mask': Uses learnable global mask to filter noisy edges (default)
        # - 'no_mask': Uses original edge weights directly (no filtering)
        self.ablation_mask = ablation_mask
        if ablation_mask == 'with_mask':
            # Learnable Global Adjacency Mask (Consensus Graph)
            # Inspired by BrainPrompt Transformer (target_model.py)
            # The model learns which connections are globally relevant, filtering noisy edges
            # This is a 116x116 learnable parameter that acts as a global filter
            self.global_adj = nn.Parameter(torch.Tensor(116, 116))
            nn.init.uniform_(self.global_adj, a=0, b=1)  # Initialize uniformly like reference
            
            # Use LayerNorm instead of BatchNorm1d since global_adj is a single parameter (116, 116)
            # not a batch. LayerNorm normalizes over features (last dimension)
            self.ln_global_adj = nn.LayerNorm(116)
            print(f"[Ablation] Mask: with_mask (Learnable Global Adjacency Mask)")
        else:
            # No mask: will use original edge weights directly
            self.global_adj = None
            self.ln_global_adj = None
            print(f"[Ablation] Mask: no_mask (No global mask, using original edge weights)")
        
        # Classification layers (Improved FC architecture)
        # Build flexible FC layers with configurable depth, activation, and normalization
        self.fc_layers = nn.ModuleList()
        self.fc_activations = nn.ModuleList()
        self.fc_batchnorms = nn.ModuleList() if fc_use_batchnorm else None
        
        # Select activation function
        if fc_activation == 'relu':
            activation_fn = nn.ReLU()
        elif fc_activation == 'elu':
            activation_fn = nn.ELU()
        elif fc_activation == 'gelu':
            activation_fn = nn.GELU()
        else:
            activation_fn = nn.ReLU()  # Default fallback
        
        # Build FC layers: pool_dim -> fc_hidden_dim -> ... -> fc_hidden_dim -> num_classes
        current_dim = pool_dim
        for i in range(fc_num_layers):
            # Hidden layer
            self.fc_layers.append(nn.Linear(current_dim, fc_hidden_dim))
            self.fc_activations.append(activation_fn)
            if fc_use_batchnorm:
                self.fc_batchnorms.append(nn.BatchNorm1d(fc_hidden_dim))
            current_dim = fc_hidden_dim
        
        # Final classification layer
        self.fc_final = nn.Linear(current_dim, num_classes)
        
        # Store configuration for forward pass
        self.fc_num_layers = fc_num_layers
        self.fc_use_batchnorm = fc_use_batchnorm
        self.fc_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        # Xavier uniform for hidden layers
        for layer in self.fc_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # More aggressive initialization for final layer (generate larger outputs)
        # gain=2.0 increases output scale to avoid very small logits
        nn.init.xavier_uniform_(self.fc_final.weight, gain=2.0)
        nn.init.normal_(self.fc_final.bias, mean=0.0, std=0.1)  # Small but non-zero bias
        
        # --- Subject-Level Prompts and Population Graph (Section 3.2) ---
        self.use_subject_prompt = use_subject_prompt
        self.text_encoder = text_encoder  # Shared Llama-encoder-1.0B (frozen)
        
        if use_subject_prompt:
            # Population Graph GNN_p: single-layer GNN for refining subject representations
            # Input: pooled graph representation (dimension depends on GAT architecture)
            # After first GAT layer with num_heads: hidden_dim * num_heads
            # After subsequent layers: hidden_dim (single head)
            # We'll determine the actual dimension dynamically in forward pass
            # For now, use a reasonable default that will be adjusted if needed
            if num_layers == 1:
                pool_input_dim = hidden_dim * num_heads  # First layer concatenates heads
            else:
                pool_input_dim = hidden_dim  # After intermediate layers, single head
            
            self.gnn_p = GATConv(
                pool_input_dim, pool_input_dim, heads=1, dropout=dropout, 
                concat=False, add_self_loops=True  # Self-loops for population graph
            )
            self.population_graph_dim = pool_input_dim
            print(f"[Subject Prompt] Population Graph enabled (GNN_p, input_dim={pool_input_dim})")
        else:
            self.gnn_p = None
            self.population_graph_dim = None
        
        # --- Disease-Level Prompts (Section 3.3) ---
        self.use_disease_prompt = use_disease_prompt
        
        if use_disease_prompt:
            # Disease prompts will be encoded once and stored
            # They are used only in loss calculation, not in forward pass
            self.disease_prompt_embeddings = None  # Will be set during initialization if text_encoder provided
            print(f"[Disease Prompt] Enabled (for L_disease loss)")
        else:
            self.disease_prompt_embeddings = None
    
    def get_mask_prompt_l1_norm(self):
        """
        Returns the L1 norm of the soft prompt for regularization.
        Useful for adding L1 regularization to the loss during training.
        
        Returns:
            torch.Tensor: L1 norm normalized by number of elements (0.0 if no soft_prompt)
        """
        if self.soft_prompt is None:
            # Return zero tensor for no_prompt mode
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.norm(self.soft_prompt, p=1) / self.soft_prompt.numel()
    
    def forward(self, batch_data, subject_prompts=None, demographics=None):
        """
        Forward pass through the GAT model.
        
        Args:
            batch_data: PyTorch Geometric Batch object containing:
                - x: Node features [Batch_Size * Num_Nodes, Input_Dim]
                - edge_index: Edge connectivity [2, Num_Edges]
                - edge_attr or edge_weight: Edge attributes [Num_Edges] or [Num_Edges, 1]
                - batch: Batch assignment for each node [Batch_Size * Num_Nodes]
        
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

        # --- Prompt Injection (Additive with semantic initialization) ---
        # Ablation study: Different modes control how prompts are injected
        # - 'no_prompt': Skip prompt injection entirely (pure GAT)
        # - 'random_prompt': Use only randomly initialized soft_prompt
        # - 'static_prompt': Use only BERT embeddings (soft_prompt is zero and non-trainable)
        # - 'full': Use BERT + trainable soft_prompt (default)
        if self.ablation_mode == 'no_prompt':
            # No prompt injection: use raw features
            pass  # x remains unchanged
        elif x.shape[0] % self.num_nodes == 0:
            num_graphs = x.shape[0] // self.num_nodes
            
            if self.ablation_mode == 'random_prompt':
                # Random prompt: use only soft_prompt (no BERT)
                base_prompt = self.soft_prompt
            elif self.ablation_mode == 'static_prompt':
                # Static prompt: use only BERT embeddings (soft_prompt is zero)
                if self.static_prompt is not None and self.prompt_projector is not None:
                    base_prompt = self.prompt_projector(self.static_prompt.to(x.device))
                else:
                    base_prompt = torch.zeros(self.num_nodes, x.shape[1], device=x.device)
            else:  # ablation_mode == 'full'
                # Full model: combine BERT and trainable soft prompt
                if self.static_prompt is not None and self.prompt_projector is not None:
                    projected = self.prompt_projector(self.static_prompt.to(x.device))
                    base_prompt = projected + self.soft_prompt
                else:
                    base_prompt = self.soft_prompt
            
            # Repeat prompt for each graph in the batch and inject
            prompt_batch = base_prompt.repeat(num_graphs, 1)
            x = x + prompt_batch
            
            # Apply normalization only if enabled
            # LayerNorm was zeroing features when std was too small
            if self.input_norm is not None:
                x = self.input_norm(x)
        else:
            # Fallback (should not occur with fixed atlas)
            pass
        
        # --- Learnable Global Adjacency Mask (Consensus Graph) - Ablation Study ---
        # Ablation modes:
        # - 'with_mask': Apply learnable global mask to filter noisy edges (default)
        # - 'no_mask': Use original edge weights directly (no filtering)
        if self.ablation_mask == 'with_mask' and self.global_adj is not None:
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
        else:
            # No mask: use original edge weights directly
            edge_attr_weighted = edge_attr
        
        # GAT Layers
        # Pass edge_attr_weighted to all layers for consistency
        # The learned global mask should be applied in all layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr_weighted)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Additional GAT layers - use edge_attr_weighted in all layers
        # Add LayerNorm and residual connections for stability (optional)
        for i, conv in enumerate(self.convs):
            residual = x  # Save residual before transformation
            
            # Apply convolution
            x = conv(x, edge_index, edge_attr=edge_attr_weighted)
            
            # Add residual connection BEFORE activation if enabled (helps with gradient flow)
            if self.use_residual:
                # Project residual if dimensions don't match (first intermediate layer)
                if i == 0 and self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                x = x + residual
            
            # Apply activation
            x = F.elu(x)
            
            # Apply LayerNorm after activation if enabled
            if self.use_layernorm_gat and self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            x = self.dropout(x)
        
        # Global Pooling - Ablation Study
        # Different pooling strategies:
        # - 'attention': Global Attention Pooling (learns importance weights, default)
        # - 'mean': Simple mean pooling (average all nodes)
        # - 'max': Max pooling (take maximum across nodes)
        weights = None
        if batch is not None:
            if self.ablation_pooling == 'attention' and self.gate_nn is not None:
                # Global Attention Pooling (GAT-LI method)
                # Manually implemented to expose attention weights for TopK Loss
                # 1. Get scores for each node: gate_nn(x) -> (N, 1)
                scores = self.gate_nn(x).view(-1, 1)  # Ensure (N, 1)
                
                # 2. Calculate weights via Softmax (per graph in batch)
                weights = softmax(scores, batch)  # (N, 1)
                
                # 3. Weighted Sum: x_pool = sum(weights * x)
                # Aggregates weighted features per graph
                x_weighted = x * weights
                x = scatter(x_weighted, batch, dim=0, reduce='add')  # (Batch, Hidden)
            elif self.ablation_pooling == 'mean':
                # Mean Pooling: Average all node features per graph
                x = scatter(x, batch, dim=0, reduce='mean')  # (Batch, Hidden)
                # Create uniform weights for compatibility (not used but may be needed for TopK Loss)
                # Weights shape should match original node count for compatibility
                num_nodes_total = batch.shape[0]
                weights = torch.ones(num_nodes_total, 1).to(x.device) / self.num_nodes
            elif self.ablation_pooling == 'max':
                # Max Pooling: Take maximum across nodes per graph
                x = scatter(x, batch, dim=0, reduce='max')  # (Batch, Hidden)
                # Create uniform weights for compatibility
                num_nodes_total = batch.shape[0]
                weights = torch.ones(num_nodes_total, 1).to(x.device) / self.num_nodes
            else:
                # Fallback to mean if unknown pooling type
                x = scatter(x, batch, dim=0, reduce='mean')
                num_nodes_total = batch.shape[0]
                weights = torch.ones(num_nodes_total, 1).to(x.device) / self.num_nodes
        else:
            # Fallback: mean pooling if no batch information
            if self.ablation_pooling == 'max':
                x = x.max(dim=0, keepdim=True)[0]
            else:  # mean or attention fallback
                x = x.mean(dim=0, keepdim=True)
            weights = torch.ones(x.shape[0], 1).to(x.device) / x.shape[0]

        # --- Subject-Level Prompts: Population Graph (Section 3.2) ---
        # Build Population Graph G_pop if subject prompts are enabled
        if self.use_subject_prompt and subject_prompts is not None and self.gnn_p is not None:
            # x is now (batch_size, pool_dim) after pooling
            batch_size = x.shape[0]
            pool_dim = x.shape[1]  # Actual dimension after pooling
            
            # M = [m1, ..., m_bz]: graph representations (already computed via pooling)
            M = x  # (batch_size, pool_dim)
            
            # Adjust GNN_p input dimension if needed (should match pool_dim)
            if pool_dim != self.population_graph_dim:
                # Recreate GNN_p with correct dimension (this should be rare)
                dropout_rate = self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.5
                self.gnn_p = GATConv(
                    pool_dim, pool_dim, heads=1, dropout=dropout_rate,
                    concat=False, add_self_loops=True
                ).to(x.device)
                self.population_graph_dim = pool_dim
            
            # Encode subject prompts if not already encoded
            if isinstance(subject_prompts[0], str):
                # Prompts are strings, need encoding
                if self.text_encoder is not None:
                    with torch.no_grad():  # Freeze encoder
                        # Encode all prompts
                        subject_embeddings = []
                        for prompt in subject_prompts:
                            if hasattr(self.text_encoder, 'encode'):
                                # LLM2Vec interface
                                emb = self.text_encoder.encode([prompt], convert_to_numpy=False, show_progress_bar=False)
                                if isinstance(emb, torch.Tensor):
                                    subject_embeddings.append(emb.squeeze(0))
                                else:
                                    subject_embeddings.append(torch.tensor(emb[0], device=x.device))
                            else:
                                # Transformers interface
                                inputs = self.text_encoder.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                                inputs = {k: v.to(x.device) for k, v in inputs.items()}
                                outputs = self.text_encoder.model(**inputs)
                                emb = outputs.last_hidden_state.mean(dim=1).squeeze()
                                subject_embeddings.append(emb)
                        subject_embeddings = torch.stack(subject_embeddings)  # (batch_size, embedding_dim)
                else:
                    print("Warning: text_encoder not provided, skipping subject prompt encoding")
                    subject_embeddings = None
            else:
                # Prompts are already encoded
                subject_embeddings = torch.stack(subject_prompts).to(x.device) if isinstance(subject_prompts, list) else subject_prompts.to(x.device)
            
            if subject_embeddings is not None:
                # Calculate similarity matrices
                # S1(i,j) = cosine_similarity(mi, mj): similarity between graph embeddings
                # S2(i,j) = cosine_similarity(Enc(psubject_i), Enc(psubject_j)): similarity between prompt embeddings
                
                # Normalize for cosine similarity
                M_norm = F.normalize(M, p=2, dim=1)  # (batch_size, hidden_dim)
                subject_emb_norm = F.normalize(subject_embeddings, p=2, dim=1)  # (batch_size, embedding_dim)
                
                # Project subject embeddings to hidden_dim if needed
                if subject_embeddings.shape[1] != M.shape[1]:
                    if not hasattr(self, 'subject_prompt_projector'):
                        self.subject_prompt_projector = nn.Linear(subject_embeddings.shape[1], M.shape[1]).to(x.device)
                        nn.init.xavier_uniform_(self.subject_prompt_projector.weight)
                    subject_embeddings = self.subject_prompt_projector(subject_embeddings)
                    subject_emb_norm = F.normalize(subject_embeddings, p=2, dim=1)
                else:
                    subject_emb_norm = F.normalize(subject_embeddings, p=2, dim=1)
                
                # Compute similarity matrices
                S1 = torch.mm(M_norm, M_norm.T)  # (batch_size, batch_size)
                S2 = torch.mm(subject_emb_norm, subject_emb_norm.T)  # (batch_size, batch_size)
                
                # Hadamard product
                S_combined = S1 * S2  # (batch_size, batch_size)
                
                # Threshold to create adjacency matrix A'
                # Use median as threshold (can be made configurable)
                # Create mask to exclude diagonal
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
                off_diagonal_values = S_combined[mask]
                if len(off_diagonal_values) > 0:
                    threshold = torch.median(off_diagonal_values)
                else:
                    threshold = torch.tensor(0.5, device=x.device)  # Fallback threshold
                
                A_prime = (S_combined > threshold).float()
                
                # Ensure self-connections (diagonal = 1)
                A_prime.fill_diagonal_(1.0)
                
                # Convert to edge_index format for PyG
                edge_index = []
                edge_weights = []
                for i in range(batch_size):
                    for j in range(batch_size):
                        if A_prime[i, j] > 0:
                            edge_index.append([i, j])
                            edge_weights.append(S_combined[i, j].item())
                
                if len(edge_index) > 0:
                    edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device).T
                    edge_weights = torch.tensor(edge_weights, dtype=torch.float, device=x.device)
                    
                    # Apply GNN_p to Population Graph
                    x = self.gnn_p(x, edge_index, edge_attr=edge_weights.unsqueeze(-1))
                    x = F.elu(x)
                    x = self.dropout(x)
                else:
                    print("Warning: No edges in Population Graph, skipping GNN_p")
        
        # Classification layers
        # Process through fully connected layers with improved architecture
        x_intermediate = x
        for i in range(self.fc_num_layers):
            x_intermediate = self.fc_layers[i](x_intermediate)
            x_intermediate = self.fc_activations[i](x_intermediate)
            if self.fc_use_batchnorm and self.fc_batchnorms is not None:
                x_intermediate = self.fc_batchnorms[i](x_intermediate)
            x_intermediate = self.fc_dropout(x_intermediate)
        
        # Final classification layer
        x = self.fc_final(x_intermediate)  # (Batch, num_classes)
        
        # Return prediction, attention weights (for TopK Loss), and intermediate features
        return x, weights, x_intermediate
