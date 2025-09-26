#!/usr/bin/env python3

"""
URL-based Fake News Prediction Script
Analyzes news URLs and predicts if they contain fake or real news
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool as gmp
from torch.utils.data import random_split
import argparse
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from datetime import datetime

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

class NewsAnalyzer:
    def __init__(self, model_type='gcn'):
        self.model_type = model_type
        self.model = None
        self.dataset = None
        
    def load_and_train_model(self, dataset_name='politifact'):
        """Load dataset and train a simple model"""
        print(f"Loading {dataset_name} dataset to train model...")
        
        # Load dataset
        self.dataset = UPFD(root='data_upfd', name=dataset_name, feature='bert')
        
        # Create model
        self.model = SimpleGNN(
            num_features=self.dataset.num_features,
            hidden_dim=128,
            num_classes=self.dataset.num_classes,
            model_type=self.model_type
        )
        
        # Quick training (in practice, this would be pre-trained)
        print("Training model (this would be pre-trained in practice)...")
        
        # Split dataset
        num_train = int(0.8 * len(self.dataset))
        num_test = len(self.dataset) - num_train
        train_dataset, test_dataset = random_split(self.dataset, [num_train, num_test])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        
        for epoch in range(3):  # Quick training
            for data in train_loader:
                optimizer.zero_grad()
                out = self.model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        
    def extract_url_features(self, url):
        """Extract features from URL and content"""
        try:
            # Get page content
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic info
            title = self._get_title(soup)
            content = self._get_content(soup)
            domain = urlparse(url).netloc
            
            return title, content, domain
            
        except Exception as e:
            print(f"Could not fetch content: {e}")
            domain = urlparse(url).netloc if url else "unknown.com"
            return "Sample Article", "Sample content", domain
    
    def _get_title(self, soup):
        """Extract title from HTML"""
        for selector in ['h1', 'title', '[property="og:title"]']:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "Unknown Title"
    
    def _get_content(self, soup):
        """Extract content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        return content[:500] if content else "No content found"
    
    def create_mock_graph(self, url, title, content, domain):
        """Create a mock graph for demonstration"""
        # Create simple features based on URL characteristics
        features = []
        
        # Domain features
        trusted_domains = ['reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com', 'thehindu.com', 'timesofindia.indiatimes.com']
        is_trusted = 1 if any(trusted in domain for trusted in trusted_domains) else 0
        
        # URL features
        url_len = len(url)
        has_https = 1 if url.startswith('https') else 0
        domain_parts = len(domain.split('.'))
        
        # Content features
        title_len = len(title.split())
        content_len = len(content.split())
        
        # Suspicious patterns
        suspicious_words = ['breaking', 'shocking', 'unbelievable', 'exclusive', 'leaked']
        suspicious_count = sum(1 for word in suspicious_words if word.lower() in title.lower())
        
        # Create base features
        base_features = [
            is_trusted, url_len/1000, has_https, domain_parts,
            title_len/10, content_len/100, suspicious_count
        ]
        
        # Pad to match dataset features (768 for BERT)
        num_features = self.dataset.num_features
        padded_features = base_features + [0] * (num_features - len(base_features))
        
        # Create mock social network (simulating users sharing the article)
        num_nodes = np.random.randint(5, 20)
        node_features = []
        
        # Article node (first node)
        node_features.append(padded_features)
        
        # User nodes (mock social media users)
        for i in range(num_nodes - 1):
            user_features = np.random.randn(num_features) * 0.1
            # Add some domain influence
            user_features[0] = is_trusted + np.random.normal(0, 0.2)
            node_features.append(user_features.tolist())
        
        # Create edges (article connected to users, some user-user connections)
        edge_list = []
        
        # Connect article (node 0) to all users
        for i in range(1, num_nodes):
            edge_list.append([0, i])
            edge_list.append([i, 0])
        
        # Add some user-user connections
        for i in range(1, num_nodes):
            if np.random.random() < 0.3:  # 30% chance of connection
                j = np.random.randint(1, num_nodes)
                if i != j:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch), num_nodes, len(edge_list)
    
    def predict(self, url):
        """Predict if news is fake or real"""
        print(f"\n📰 Analyzing news article: {url}")
        
        # Extract features
        title, content, domain = self.extract_url_features(url)
        
        print("🔍 In a real system, this would:")
        print("   1. Scrape article content")
        print("   2. Fetch social media sharing data")
        print("   3. Build interaction graph")
        print("   4. Extract BERT text features")
        print("   5. Get user profile features")
        
        # Create mock graph
        graph, num_nodes, num_edges = self.create_mock_graph(url, title, content, domain)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(graph)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        return {
            'url': url,
            'title': title,
            'domain': domain,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0][0].item(),
                'fake': probabilities[0][1].item()
            },
            'graph_stats': {
                'nodes': num_nodes,
                'edges': num_edges
            }
        }

def main():
    parser = argparse.ArgumentParser(description='GNN-based Fake News Detection')
    parser.add_argument('--url', type=str, required=True, help='News article URL to analyze')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sage', 'gat'], help='GNN model type')
    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'], help='Dataset to use')
    
    args = parser.parse_args()
    
    print("🤖 GNN-Based Fake News Detection")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NewsAnalyzer(model_type=args.model)
    
    # Load and train model
    analyzer.load_and_train_model(args.dataset)
    
    # Analyze URL
    result = analyzer.predict(args.url)
    
    # Display results
    print(f"\n📊 PREDICTION RESULTS")
    print("=" * 30)
    print(f"🔗 URL: {result['url']}")
    print(f"🧠 Model: {args.model.upper()}")
    print(f"📈 Dataset: {args.dataset}")
    print(f"👥 Graph nodes: {result['graph_stats']['nodes']}")
    print(f"🔗 Graph edges: {result['graph_stats']['edges']}")
    
    label = "FAKE" if result['prediction'] == 1 else "REAL"
    emoji = "❌" if label == "FAKE" else "✅"
    
    print(f"\n{emoji} PREDICTION: {label}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {result['probabilities']['real']:.2%}")
    print(f"   Fake: {result['probabilities']['fake']:.2%}")
    
    print(f"\n⚠️  NOTE: This is a demonstration using dummy data.")
    print("   Real implementation requires social media APIs and web scraping.")

if __name__ == '__main__':
    main()