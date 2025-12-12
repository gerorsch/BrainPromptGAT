# BrainPromptGAT

Graph Attention Network (GAT) implementation for the BrainPrompt framework, adapting the original Transformer architecture to graph neural networks for brain network analysis.

## Academic Context

This project was developed for the **Computer Vision** course (Visão Computacional) of the **Bachelor's Degree in Computer Science** (Bacharelado em Ciência da Computação) at **UFRPE** (Universidade Federal Rural de Pernambuco).

## Project Structure

```
BrainPromptGAT/
├── graph_utils.py              # Utilities for matrix→graph conversion
├── gnn_baseline_model.py       # GAT baseline model architecture
├── gnn_baseline_train.py       # Baseline training script
├── braingnn_losses.py          # Loss functions (TopK, Consistency)
├── focal_loss_utils.py         # Focal Loss implementation
├── setting.py                  # Configuration and arguments
├── utils.py                    # Utility functions
├── main.py                     # Entry point with Optuna support
├── prepare_data.py             # Data preparation utilities
├── setup_data.py               # Data setup script
├── generate_prompts.py         # Prompt generation (if applicable)
├── modify_data_process.py      # Data processing modifications
├── data/
│   └── roi_bert_embeddings.pt  # Pre-trained BERT embeddings for ROIs
└── README.md                   # This file
```

## Dependencies

See `requirements.txt` in the project root.

Main dependencies:
- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn
- Optuna (for hyperparameter optimization)

## Prerequisites

1. **Processed Data**: Data should be processed by `BrainPrompt/data_process.py`
   - Code supports both vector and matrix formats
   - If matrix format doesn't exist, automatically converts from vector format

2. **Prepare Data** (optional but recommended):
   ```bash
   # Option 1: Modify data_process.py to save in matrix format
   python BrainPromptGAT/modify_data_process.py
   # Then run BrainPrompt/data_process.py
   
   # Option 2: Convert existing data from vector to matrix
   python BrainPromptGAT/prepare_data.py --site NYU --convert
   
   # Option 3: Check which data files exist
   python BrainPromptGAT/prepare_data.py --site NYU
   ```

## Usage

### Step 0: Download ABIDE Data (First Time)

**IMPORTANT**: Before processing data, you need to download the ABIDE dataset.

```bash
# Option 1: Use helper script
cd BrainPromptGAT
python setup_data.py --download

# Option 2: Download manually
cd BrainPrompt
python ABIDE_download.py
```

**Note**: Download may take considerable time (several GB) and requires internet connection.

### Prepare Data (First Time)

```bash
# 1. Check if data exists
python prepare_data.py --site NYU

# 2a. Option A: Modify data_process.py to save matrix format
python modify_data_process.py
cd ../BrainPrompt
python data_process.py  # This will generate data in both formats

# 2b. Option B: Convert existing data
python prepare_data.py --site NYU --convert
```

### Baseline Training

```bash
python main.py --model gnn_baseline_model --site NYU --gat_heads 4 --gat_hidden_dim 32 --gat_num_layers 2
```

### Main Parameters

- `--model`: Model to use (`gnn_baseline_model`, `gnn_target_model`, etc.)
- `--site`: Target site (`NYU`, `UCLA`, `UM`, `USM`)
- `--gat_heads`: Number of attention heads (default: 4)
- `--gat_hidden_dim`: Hidden dimension for GAT (default: 32)
- `--gat_num_layers`: Number of GAT layers (default: 2)
- `--temporal_strategy`: Temporal strategy (`mean` or `separate`, default: `mean`)
- `--edge_threshold`: Threshold for graph sparsification (default: 0.0 = dense graph)
- `--lr`: Learning rate (default: 0.001)
- `--epoch_cf`: Maximum epochs (default: 200)
- `--accumulation_steps`: Gradient accumulation steps (default: 32)
- `--focal_gamma`: Focal Loss gamma parameter (default: 1.0)
- `--lambda_tpk`: TopK Loss weight (default: 0.1)

## Differences from Original BrainPrompt

1. **Architecture**: GAT instead of Transformer
2. **Data Format**: Correlation matrices (116x116) instead of vectors (6670)
3. **Processing**: Graphs instead of sequences
4. **Temporal**: Temporal mean or separate window processing

## Implementation Status

- [x] `graph_utils.py` - Matrix→graph conversion
- [x] `gnn_baseline_model.py` - GAT baseline model
- [x] `gnn_baseline_train.py` - Baseline training
- [x] `braingnn_losses.py` - Loss functions (TopK, Consistency)
- [x] `focal_loss_utils.py` - Focal Loss implementation
- [x] `setting.py` - Configuration
- [x] `utils.py` - Utilities
- [x] `main.py` - Entry point
- [ ] `gnn_target_model.py` - Model with prompts (future work)
- [ ] `gnn_source1_train.py` - Mask prompt training (future work)
- [ ] `gnn_source2_train.py` - LoRA training (future work)
- [ ] `gnn_target_train.py` - Target training (future work)

## Key Features

- **Semantic Prompt Injection**: Pre-trained BERT embeddings for each ROI
- **Learnable Global Adjacency Mask**: Consensus graph that filters noisy connections
- **Multi-layer GAT Encoder**: Graph attention with edge attributes
- **Global Attention Pooling**: Aggregates node features to graph-level
- **Comprehensive Loss Functions**: Focal Loss, TopK Loss, Diversity Loss, L1 regularization
- **5-fold Cross-Validation**: Robust evaluation
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive LR reduction

## Notes

- Baseline model is functional and ready for testing
- Models with prompt learning will be implemented in subsequent phases
- Default temporal strategy is `mean` (average of 25 windows)
- Debug logging is available (enabled only in first iteration of first epoch)

