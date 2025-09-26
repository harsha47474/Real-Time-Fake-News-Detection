# Quick Start Guide

This guide shows how to quickly run the GNN-FakeNews project using the built-in PyTorch Geometric UPFD dataset.

## Prerequisites

All required dependencies have been installed:
- Python 3.13+
- PyTorch 2.8.0
- PyTorch Geometric 2.6.1
- Required extensions (torch_sparse, torch_scatter, etc.)

## Running the Models

### Option 1: Run a Single Model

```bash
# Run GCN on Politifact dataset
python test_upfd.py --dataset politifact --model gcn --epochs 10

# Run SAGE on Gossipcop dataset  
python test_upfd.py --dataset gossipcop --model sage --epochs 10

# Run GAT on Politifact dataset
python test_upfd.py --dataset politifact --model gat --epochs 10
```

### Option 2: Run All Models

```bash
# Run all model combinations (GCN, SAGE, GAT on both datasets)
python run_all_models.py
```

## Available Options

- **Datasets**: `politifact`, `gossipcop`
- **Models**: `gcn`, `sage`, `gat`  
- **Features**: `bert` (default), `profile`, `spacy`, `content`
- **Other parameters**: `--epochs`, `--lr`, `--batch_size`, `--hidden_dim`

## Example Results

The models will automatically:
1. Download the UPFD dataset (first run only)
2. Train the specified GNN model
3. Evaluate on test set
4. Display accuracy, F1-score, precision, and recall

## Dataset Information

- **Politifact**: 62 graphs, 768 features (BERT), 2 classes
- **Gossipcop**: 1092 graphs, 768 features (BERT), 2 classes

The datasets are automatically downloaded from the official PyTorch Geometric repository.