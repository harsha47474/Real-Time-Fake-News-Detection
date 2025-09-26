#!/usr/bin/env python3

"""
Train fake news detection model using Hugging Face dataset
Dataset: https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
from tqdm import tqdm
import pickle

class FakeNewsDataset(Dataset):
    """Custom dataset for fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTFakeNewsClassifier(nn.Module):
    """BERT-based fake news classifier"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(BERTFakeNewsClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class FakeNewsTrainer:
    """Trainer class for fake news detection model"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_huggingface_dataset(self):
        """Load the Hugging Face dataset"""
        print("📥 Loading Hugging Face dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("Pulk17/Fake-News-Detection-dataset")
            
            # Convert to pandas DataFrame
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no train split, use the whole dataset
                df = dataset.to_pandas()
            
            print(f"✅ Dataset loaded: {len(df)} samples")
            print(f"📊 Columns: {df.columns.tolist()}")
            print(f"📈 Dataset shape: {df.shape}")
            
            # Display first few rows
            print("\n📋 Sample data:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("📝 Trying alternative approach...")
            
            # Alternative: Load from CSV if available
            return self.load_alternative_dataset()
    
    def load_alternative_dataset(self):
        """Load alternative dataset if Hugging Face fails"""
        print("📥 Loading alternative fake news dataset...")
        
        # Create a sample dataset for demonstration
        fake_news_data = {
            'text': [
                "BREAKING: Scientists discover miracle cure that doctors don't want you to know!",
                "SHOCKING: Government covers up alien landing in Nevada desert!",
                "Reuters reports on latest economic developments in global markets",
                "BBC News: Climate change summit reaches historic agreement",
                "UNBELIEVABLE: This one trick will make you rich overnight!",
                "Associated Press: Election results confirmed by independent observers",
                "CNN: Breaking news on international diplomatic relations",
                "LEAKED: Secret documents reveal government conspiracy!",
                "The Guardian: Analysis of current political situation",
                "New York Times: In-depth investigation reveals corporate fraud"
            ],
            'label': [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]  # 1 = fake, 0 = real
        }
        
        df = pd.DataFrame(fake_news_data)
        print(f"✅ Alternative dataset created: {len(df)} samples")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        print("🔄 Preprocessing data...")
        
        # Handle different column names
        text_columns = ['text', 'title', 'content', 'news', 'article']
        label_columns = ['label', 'fake', 'class', 'target']
        
        text_col = None
        label_col = None
        
        # Find text column
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        # Find label column
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None or label_col is None:
            print(f"❌ Could not find text or label columns")
            print(f"Available columns: {df.columns.tolist()}")
            return None, None
        
        print(f"📝 Using text column: {text_col}")
        print(f"🏷️  Using label column: {label_col}")
        
        # Extract texts and labels
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = df[label_col].tolist()
        
        # Convert labels to binary if needed
        unique_labels = set(labels)
        print(f"🏷️  Unique labels: {unique_labels}")
        
        if len(unique_labels) > 2:
            print("⚠️  More than 2 labels found, converting to binary...")
            # Convert to binary (assuming higher values = fake)
            median_label = np.median(labels)
            labels = [1 if label > median_label else 0 for label in labels]
        
        # Ensure labels are 0 and 1
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        if not all(label in [0, 1] for label in labels):
            labels = [label_mapping[label] for label in labels]
        
        print(f"✅ Preprocessed {len(texts)} samples")
        print(f"📊 Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def create_data_loaders(self, texts, labels, test_size=0.2, batch_size=16):
        """Create train and test data loaders"""
        print("📦 Creating data loaders...")
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = FakeNewsDataset(test_texts, test_labels, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Train samples: {len(train_dataset)}")
        print(f"✅ Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train_model(self, train_loader, test_loader, epochs=3, learning_rate=2e-5):
        """Train the BERT model"""
        print("🏋️ Training BERT model...")
        
        # Initialize model
        model = BERTFakeNewsClassifier(self.model_name)
        model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}'
                })
            
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions / total_predictions
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Evaluation phase
            test_accuracy = self.evaluate_model(model, test_loader)
            test_accuracies.append(test_accuracy)
            
            print(f"📊 Epoch {epoch + 1} Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Train Accuracy: {train_accuracy:.4f}")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model_path = 'models/bert_fake_news_model.pth'
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': self.model_name,
            'tokenizer': self.tokenizer,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_losses': train_losses
        }, model_path)
        
        print(f"✅ Model saved to: {model_path}")
        
        # Plot training history
        self.plot_training_history(train_losses, train_accuracies, test_accuracies)
        
        return model
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def plot_training_history(self, train_losses, train_accuracies, test_accuracies):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
        
        print("📈 Training history saved to: models/training_history.png")

def main():
    """Main training function"""
    print("🚀 BERT Fake News Detection Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = FakeNewsTrainer()
    
    # Load dataset
    df = trainer.load_huggingface_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Preprocess data
    texts, labels = trainer.preprocess_data(df)
    if texts is None:
        print("❌ Failed to preprocess data")
        return
    
    # Create data loaders
    train_loader, test_loader = trainer.create_data_loaders(texts, labels)
    
    # Train model
    model = trainer.train_model(train_loader, test_loader, epochs=3)
    
    print("\n✅ Training completed!")
    print("📁 Model saved to: models/bert_fake_news_model.pth")
    print("📈 Training history: models/training_history.png")
    
    print("\n💡 Next steps:")
    print("   1. Use the trained model in your web interface")
    print("   2. Test with real news articles")
    print("   3. Fine-tune hyperparameters if needed")

if __name__ == '__main__':
    # Install required packages
    try:
        import datasets
        import transformers
        import matplotlib
        import seaborn
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install datasets transformers matplotlib seaborn")
    
    main()