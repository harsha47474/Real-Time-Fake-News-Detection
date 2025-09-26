#!/usr/bin/env python3

"""
Train models using the Hugging Face dataset
Dataset: https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import os
from datasets import load_dataset
import time

class HuggingFaceTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        print("🤖 Hugging Face Dataset Trainer initialized")
    
    def load_dataset(self):
        print("📥 Loading Hugging Face dataset: Pulk17/Fake-News-Detection-dataset")
        
        try:
            dataset = load_dataset("Pulk17/Fake-News-Detection-dataset")
            print("✅ Dataset loaded successfully!")
            
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
            
            print(f"📈 Dataset shape: {df.shape}")
            print(f"📊 Columns: {df.columns.tolist()}")
            print("\n📋 Sample data:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        print("🔄 Preprocessing data...")
        
        # Find text and label columns
        text_col = None
        label_col = None
        
        for col in ['text', 'title', 'content', 'news']:
            if col in df.columns:
                text_col = col
                break
        
        for col in ['label', 'fake', 'class', 'target']:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None or label_col is None:
            print(f"Available columns: {df.columns.tolist()}")
            if len(df.columns) >= 2:
                text_col = df.columns[0]
                label_col = df.columns[1]
            else:
                return None, None
        
        print(f"📝 Using text column: {text_col}")
        print(f"🏷️  Using label column: {label_col}")
        
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = df[label_col].tolist()
        
        # Convert labels to binary
        unique_labels = sorted(set(labels))
        print(f"🏷️  Unique labels: {unique_labels}")
        
        if len(unique_labels) == 2:
            if not set(unique_labels) == {0, 1}:
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                labels = [label_map[label] for label in labels]
        
        print(f"✅ Preprocessed {len(texts)} samples")
        print(f"📊 Real: {sum(1 for l in labels if l == 0)}, Fake: {sum(1 for l in labels if l == 1)}")
        
        return texts, labels
    
    def train_models(self, texts, labels):
        print("🏋️ Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("🔄 Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train Logistic Regression
        print("📈 Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_vec, y_train)
        lr_pred = lr_model.predict(X_test_vec)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"✅ Logistic Regression Accuracy: {lr_acc:.1%}")
        
        # Train Random Forest
        print("🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_vec, y_train)
        rf_pred = rf_model.predict(X_test_vec)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"✅ Random Forest Accuracy: {rf_acc:.1%}")
        
        # Save models
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'logistic_regression': lr_model,
            'random_forest': rf_model,
            'accuracies': {
                'logistic_regression': lr_acc,
                'random_forest': rf_acc
            }
        }
        
        with open('models/huggingface_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Models saved to: models/huggingface_models.pkl")
        
        return lr_acc, rf_acc

def main():
    print("🤖 Training with Hugging Face Dataset")
    print("=" * 50)
    
    start_time = time.time()
    
    trainer = HuggingFaceTrainer()
    
    # Load dataset
    df = trainer.load_dataset()
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Preprocess
    texts, labels = trainer.preprocess_data(df)
    if texts is None:
        print("❌ Failed to preprocess data")
        return
    
    # Train
    lr_acc, rf_acc = trainer.train_models(texts, labels)
    
    end_time = time.time()
    
    print(f"\n🎉 Training completed in {end_time - start_time:.1f} seconds!")
    print(f"📊 Logistic Regression: {lr_acc:.1%}")
    print(f"📊 Random Forest: {rf_acc:.1%}")

if __name__ == '__main__':
    main()