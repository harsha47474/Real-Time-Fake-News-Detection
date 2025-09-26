#!/usr/bin/env python3

"""
Setup script to create a pre-trained model for real_time_detector
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from test_upfd import SimpleGNN
import os
from tqdm import tqdm

def create_pretrained_model():
    """Create and save a pre-trained model"""
    print("🤖 Creating pre-trained model for real-time detector...")
    
    # Load dataset
    dataset = UPFD(root='data_upfd', name='gossipcop', feature='bert')
    print(f"📊 Dataset loaded: {len(dataset)} graphs, {dataset.num_features} features")
    
    # Create model
    model = SimpleGNN(
        num_features=dataset.num_features,
        hidden_dim=128,
        num_classes=dataset.num_classes,
        model_type='gcn'
    )
    
    # Split dataset
    num_train = int(0.8 * len(dataset))
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("🏋️ Training model...")
    model.train()
    
    # Quick training (10 epochs for demo)
    for epoch in tqdm(range(10), desc="Training"):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    accuracy = correct / total
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = 'models/real_time_detector_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_features': dataset.num_features,
        'num_classes': dataset.num_classes,
        'hidden_dim': 128,
        'model_type': 'gcn',
        'accuracy': accuracy,
        'dataset': 'gossipcop'
    }, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

if __name__ == '__main__':
    create_pretrained_model()