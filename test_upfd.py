#!/usr/bin/env python3

"""
Test script to run GNN models using the built-in PyTorch Geometric UPFD dataset
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool as gmp
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import argparse
from tqdm import tqdm
import time

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_classes=2, model_type='gcn'):
        super(SimpleGNN, self).__init__()
        self.model_type = model_type
        
        if model_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_dim)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(num_features, hidden_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_dim)
        
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution
        x = F.relu(self.conv1(x, edge_index))
        
        # Global pooling
        x = gmp(x, batch)
        
        # Classification
        x = F.log_softmax(self.classifier(x), dim=-1)
        
        return x

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1, precision, recall, total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--feature', type=str, default='bert', choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset with {args.feature} features...")
    try:
        dataset = UPFD(root='data_upfd', name=args.dataset, feature=args.feature)
        print(f"Dataset loaded successfully!")
        print(f"Number of graphs: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")
        
        # Split dataset
        num_train = int(0.6 * len(dataset))
        num_val = int(0.2 * len(dataset))
        num_test = len(dataset) - num_train - num_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [num_train, num_val, num_test]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = SimpleGNN(
            num_features=dataset.num_features,
            hidden_dim=args.hidden_dim,
            num_classes=dataset.num_classes,
            model_type=args.model
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"\nTraining {args.model.upper()} model...")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Training loop
        best_val_acc = 0
        for epoch in tqdm(range(args.epochs)):
            model.train()
            total_loss = 0
            
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            val_acc, val_f1, val_precision, val_recall, val_loss = evaluate_model(model, val_loader, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss: {total_loss/len(train_loader):.4f}, "
                      f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Final evaluation
        test_acc, test_f1, test_precision, test_recall, test_loss = evaluate_model(model, test_loader, device)
        
        print(f"\nFinal Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("The dataset will be downloaded automatically on first run.")
        return

if __name__ == '__main__':
    main()