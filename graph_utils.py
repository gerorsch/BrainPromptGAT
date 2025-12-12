"""
Graph Utilities for BrainPromptGAT

This module provides utilities for converting functional connectivity matrices
(correlation matrices) to graph format compatible with PyTorch Geometric.

Key functions:
- matrix_to_edge_index: Converts adjacency matrix to edge_index format
- create_graph_data_from_matrix: Creates PyTorch Geometric Data object
- temporal_graphs_to_batch: Processes temporal windows and creates graph batches
- load_matrix_data: Loads data in matrix format with fallback to vector format
"""
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, remove_self_loops


def matrix_to_edge_index(adjacency_matrix, threshold=None):
    """
    Convert adjacency matrix to edge_index format for PyTorch Geometric.
    
    NOTE: This function expects an adjacency matrix with absolute values (already processed).
    The threshold is applied only to the graph structure.
    
    Args:
        adjacency_matrix: Tensor of shape (num_nodes, num_nodes) or (batch, num_nodes, num_nodes)
                         Should contain absolute values (already processed)
        threshold: Optional value to sparsify the graph. If None, keeps all connections (dense graph)
    
    Returns:
        edge_index: Tensor of shape (2, num_edges) - edge connectivity
        edge_weight: Tensor of shape (num_edges,) - edge weights
    """
    if adjacency_matrix.dim() == 3:
        # Batch of graphs - process each separately
        batch_size = adjacency_matrix.shape[0]
        edge_indices = []
        edge_weights = []
        
        for i in range(batch_size):
            ei, ew = matrix_to_edge_index(adjacency_matrix[i], threshold)
            edge_indices.append(ei)
            edge_weights.append(ew)
        
        return edge_indices, edge_weights
    
    # Ensure adjacency matrix has absolute values
    # (should already be processed, but ensure for safety)
    adjacency_matrix = torch.abs(adjacency_matrix)
    
    # Apply threshold only if specified
    # When threshold=None, keep all connections (dense graph) to avoid losing information
    if threshold is not None:
        mask = adjacency_matrix >= threshold
        # Keep diagonal (auto-correlation = 1.0)
        diag_mask = torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device, dtype=torch.bool)
        mask = mask | diag_mask
        adjacency_matrix = adjacency_matrix * mask.float()
    # If threshold=None, don't apply mask - keep all connections
    
    # Convert to sparse format
    edge_index, edge_weight = dense_to_sparse(adjacency_matrix)
    
    # Remove self-loops (optional - depends on application)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    
    return edge_index, edge_weight


def create_graph_data_from_matrix(adjacency_matrix, node_features=None, threshold=None):
    """
    Create PyTorch Geometric Data object from adjacency matrix.
    
    CORRECTION BASED ON GAT-LI:
    - Features: ORIGINAL correlation matrix (preserves negative values)
    - Adjacency: absolute value of matrix (for graph structure)
    
    This design allows the model to learn about negative correlations in features,
    while using only positive values for graph structure.
    
    Args:
        adjacency_matrix: Tensor of shape (num_nodes, num_nodes) - correlation matrix
        node_features: Tensor of shape (num_nodes, feature_dim) - node features
                     If None, uses the ORIGINAL matrix row as feature (without abs)
        threshold: Optional value to sparsify (applied only to adjacency)
    
    Returns:
        data: PyTorch Geometric Data object
    """
    if isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = torch.from_numpy(adjacency_matrix).float()
    
    num_nodes = adjacency_matrix.shape[0]
    
    # If no node features, use ORIGINAL matrix row as feature
    # IMPORTANT: Features preserve negative values (negative correlations)
    if node_features is None:
        # GAT-LI uses original matrix for features (not absolute value)
        node_features = adjacency_matrix.clone()  # Each node gets its row (116 features)
    elif isinstance(node_features, np.ndarray):
        node_features = torch.from_numpy(node_features).float()
        # Don't apply abs here - preserve original values
    
    # Adjacency: use absolute value for graph structure
    # This allows the model to learn about negative correlations in features,
    # but uses only positive values for graph structure
    adj_for_structure = torch.abs(adjacency_matrix)
    
    # Convert to edge_index using adjacency with absolute value
    edge_index, edge_weight = matrix_to_edge_index(adj_for_structure, threshold)
    
    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_weight  # Compatibility with layers that expect edge_attr
    )
    
    return data


def batch_graphs_from_matrices(adjacency_matrices, node_features_list=None, threshold=None):
    """
    Cria batch de grafos a partir de múltiplas matrizes de adjacência.
    
    Args:
        adjacency_matrices: Tensor de shape (batch_size, num_nodes, num_nodes)
        node_features_list: Lista de tensores de features (opcional)
        threshold: Valor opcional para esparsificar
    
    Returns:
        batch: Objeto Batch do PyTorch Geometric
    """
    if isinstance(adjacency_matrices, np.ndarray):
        adjacency_matrices = torch.from_numpy(adjacency_matrices).float()
    
    data_list = []
    batch_size = adjacency_matrices.shape[0]
    
    for i in range(batch_size):
        node_feat = node_features_list[i] if node_features_list is not None else None
        data = create_graph_data_from_matrix(
            adjacency_matrices[i], 
            node_feat, 
            threshold
        )
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list)
    return batch


def temporal_graphs_to_batch(adjacency_matrices, node_features_list=None, threshold=None, 
                             temporal_strategy='mean', window_mask=None):
    """
    Process multiple temporal windows of graphs.
    
    CORRECTION BASED ON GAT-LI:
    - Features: temporal mean of ORIGINAL matrix (preserves negative values)
    - Adjacency: temporal mean of absolute value (for graph structure)
    
    This design preserves important information about negative correlations in features
    while using only positive values for graph structure.
    
    Args:
        adjacency_matrices: Tensor of shape (batch_size, num_windows, num_nodes, num_nodes)
                           Original correlation matrix (may have negative values)
        node_features_list: List of features (optional)
        threshold: Optional value to sparsify (applied only to adjacency)
        temporal_strategy: 'mean' (temporal mean) or 'separate' (process separately)
        window_mask: Optional temporal mask (True = valid window, False = padding)
                    Shape: (batch_size, num_windows) or (batch_size, num_windows, num_windows)
    
    Returns:
        If temporal_strategy='mean': Single batch with averaged graph
        If temporal_strategy='separate': List of batches, one per temporal window
    """
    if isinstance(adjacency_matrices, np.ndarray):
        adjacency_matrices = torch.from_numpy(adjacency_matrices).float()
    
    batch_size, num_windows, num_nodes, _ = adjacency_matrices.shape
    
    # Optional temporal mask (True = valid window)
    if window_mask is not None:
        if isinstance(window_mask, np.ndarray):
            window_mask = torch.from_numpy(window_mask).bool()
        # Reduce 3D masks (25x25) to per-window indicator if coming from BrainPrompt
        if window_mask.dim() == 3:
            window_mask = window_mask.any(dim=2)
        # Ensure shape (batch, num_windows)
        assert window_mask.shape[0] == batch_size and window_mask.shape[1] == num_windows, \
            f"window_mask shape {window_mask.shape} doesn't match {adjacency_matrices.shape}"
        # Align device with adjacency_matrices
        window_mask = window_mask.to(adjacency_matrices.device)
    else:
        window_mask = torch.ones((batch_size, num_windows), dtype=torch.bool, device=adjacency_matrices.device)
    
    if temporal_strategy == 'mean':
        # Calculate temporal mean weighted by mask
        # Structure: mean of absolute value (for graph structure, only positive values)
        # Features: ORIGINAL mean (preserves negative correlations - important information!)
        mask = window_mask.float().unsqueeze(-1).unsqueeze(-1)  # (batch, num_windows, 1, 1)
        # For graph structure: use abs() (only positive values for edge_index)
        weighted_sum_abs = (adjacency_matrices.abs() * mask).sum(dim=1)
        # For features: preserve original values (including negatives)
        weighted_sum_feat = (adjacency_matrices * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        mean_structure = weighted_sum_abs / counts  # For edge_index (abs)
        mean_features = weighted_sum_feat / counts  # For features (original, preserves negatives)
        
        # create_graph_data_from_matrix:
        # - uses adjacency = abs(...) internally for edge_index (structure)
        # - uses features = node_features (here we pass mean_features with original values)
        return batch_graphs_from_matrices(mean_structure, [feat for feat in mean_features], threshold)

    elif temporal_strategy == 'separate':
        # Process each window separately and aggregate
        # For now, process each window and average embeddings
        # Future: Implement more sophisticated temporal processing
        mask = window_mask.float().unsqueeze(-1).unsqueeze(-1)
        weighted_sum = (adjacency_matrices * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        mean_correlation = weighted_sum / counts
        return batch_graphs_from_matrices(mean_correlation, node_features_list, threshold)
    
    else:
        raise ValueError(f"Unknown temporal_strategy: {temporal_strategy}")


def vector_to_matrix(vector_data, num_nodes=116):
    """
    Converte dados de formato vetor (6670 features) para formato matriz (116x116).
    
    Args:
        vector_data: Tensor/array de shape (N, 25, 6670) ou (N, 6670)
        num_nodes: Número de nós (ROIs). Default: 116
    
    Returns:
        matrix_data: Tensor de shape (N, 25, 116, 116) ou (N, 116, 116)
    """
    if isinstance(vector_data, np.ndarray):
        vector_data = torch.from_numpy(vector_data).float()
    
    # Obter índices da parte superior triangular
    row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    
    # Garantir que vector_data seja 3D
    if vector_data.dim() == 2:
        vector_data = vector_data.unsqueeze(1)  # (N, 1, 6670)
    
    batch_size, num_windows, num_features = vector_data.shape
    
    # Verificar número de features
    expected_features = (num_nodes * (num_nodes - 1)) // 2
    assert num_features == expected_features, \
        f"Esperado {expected_features} features, mas recebeu {num_features}"
    
    # Inicializar matriz de adjacência
    # Se vector_data não tem device (CPU tensor), usar CPU
    device = vector_data.device if hasattr(vector_data, 'device') else None
    adjacency = torch.zeros(
        batch_size, num_windows, num_nodes, num_nodes,
        dtype=vector_data.dtype
    )
    if device is not None:
        adjacency = adjacency.to(device)
    
    # Preencher diagonal (auto-correlação = 1.0)
    diag_indices = torch.arange(num_nodes)
    if device is not None:
        diag_indices = diag_indices.to(device)
    adjacency[:, :, diag_indices, diag_indices] = 1.0
    
    # Preencher parte superior triangular
    for b in range(batch_size):
        for t in range(num_windows):
            triu_values = vector_data[b, t, :]  # (6670,)
            adjacency[b, t, row_indices, col_indices] = triu_values
            
            # Tornar simétrica
            adjacency[b, t] = adjacency[b, t] + adjacency[b, t].T - torch.diag(
                torch.diag(adjacency[b, t])
            )
    
    # Sempre manter dimensão temporal (mesmo que seja 1)
    # Isso garante formato consistente (N, 25, 116, 116) ou (N, 1, 116, 116)
    return adjacency


def load_matrix_data(path_data, path_mask=None, fallback_to_vector=True):
    """
    Carrega dados em formato de matriz (116x116), com fallback para formato vetor.
    
    Args:
        path_data: Caminho para arquivo .npy (matriz ou vetor)
        path_mask: Caminho para máscara temporal (opcional)
        fallback_to_vector: Se True, tenta carregar formato vetor se matriz não existir
    
    Returns:
        X: Tensor de matrizes de correlação (N, 25, 116, 116)
        mask: Tensor de máscaras (se fornecido)
    """
    import os
    
    # Tentar carregar formato matriz primeiro
    if os.path.exists(path_data):
        X = np.load(path_data)
        X = torch.from_numpy(X).float()
        
        # Verificar se já está em formato matriz
        if X.dim() == 4 and X.shape[-1] == 116:
            # Já está em formato matriz (N, 25, 116, 116)
            pass
        elif X.dim() == 3 and X.shape[-1] == 116:
            # Formato (N, 116, 116) - adicionar dimensão temporal
            X = X.unsqueeze(1)  # (N, 1, 116, 116)
        else:
            # Provavelmente formato vetor - converter
            X = vector_to_matrix(X)
    elif fallback_to_vector:
        # Tentar carregar formato vetor como fallback
        # Tenta diferentes padrões de nome
        possible_paths = [
            path_data.replace('_X_matrix.npy', '_X1.npy'),
            path_data.replace('_15_site_X_matrix.npy', '_15_site_X1.npy'),
            path_data.replace('X_matrix.npy', 'X1.npy'),
            # Tentar também sem o prefixo do site
            os.path.join(os.path.dirname(path_data), os.path.basename(path_data).replace('_X_matrix.npy', '_X1.npy')),
        ]
        
        # Remover duplicatas mantendo ordem
        seen = set()
        unique_paths = []
        for p in possible_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        possible_paths = unique_paths
        
        X_vector = None
        found_path = None
        for vector_path in possible_paths:
            if os.path.exists(vector_path):
                print(f"Arquivo de matriz não encontrado. Convertendo de formato vetor: {vector_path}")
                X_vector = np.load(vector_path)
                found_path = vector_path
                break
        
        if X_vector is not None:
            X = vector_to_matrix(X_vector)
            print(f"✓ Conversão bem-sucedida! Shape final: {X.shape}")
        else:
            # Mensagem de erro mais útil
            error_msg = (
                f"\n{'='*60}\n"
                f"ERRO: Arquivos de dados não encontrados!\n"
                f"{'='*60}\n"
                f"Arquivo esperado (matriz): {path_data}\n"
                f"\nTentou também (vetor):\n"
            )
            for i, p in enumerate(possible_paths, 1):
                error_msg += f"  {i}. {p} {'(existe)' if os.path.exists(p) else '(não existe)'}\n"
            
            error_msg += (
                f"\n{'='*60}\n"
                f"SOLUÇÃO:\n"
                f"{'='*60}\n"
                f"1. Execute o pré-processamento primeiro:\n"
                f"   cd BrainPrompt\n"
                f"   python data_process.py\n"
                f"\n"
                f"2. Ou converta dados existentes:\n"
                f"   cd BrainPromptGAT\n"
                f"   python prepare_data.py --site {os.path.basename(os.path.dirname(path_data))} --convert\n"
                f"\n"
                f"3. Ou verifique se os dados estão em outro local\n"
                f"{'='*60}\n"
            )
            raise FileNotFoundError(error_msg)
    else:
        raise FileNotFoundError(f"Arquivo não encontrado: {path_data}")
    
    mask = None
    if path_mask is not None and os.path.exists(path_mask):
        mask = np.load(path_mask)
        mask = torch.from_numpy(mask).bool()
    
    return X, mask
