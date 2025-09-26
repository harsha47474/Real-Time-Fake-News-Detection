#!/usr/bin/env python3

"""
Script to run all available GNN models on both datasets
"""

import subprocess
import sys

def run_model(dataset, model, epochs=5):
    """Run a specific model configuration"""
    print(f"\n{'='*60}")
    print(f"Running {model.upper()} on {dataset} dataset")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "test_upfd.py",
        "--dataset", dataset,
        "--model", model,
        "--epochs", str(epochs)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error running {model} on {dataset}:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"Timeout running {model} on {dataset}")
    except Exception as e:
        print(f"Exception running {model} on {dataset}: {e}")

def main():
    """Run all model combinations"""
    datasets = ['politifact', 'gossipcop']
    models = ['gcn', 'sage', 'gat']
    epochs = 5
    
    print("GNN-FakeNews Model Evaluation")
    print("This will run all GNN models on both datasets")
    print(f"Training for {epochs} epochs each")
    
    for dataset in datasets:
        for model in models:
            run_model(dataset, model, epochs)
    
    print(f"\n{'='*60}")
    print("All model evaluations completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()