#!/usr/bin/env python3

"""
BERT-based Fake News Detector using trained model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import pickle

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

class BERTFakeNewsDetector:
    """BERT-based fake news detector for URLs"""
    
    def __init__(self, model_path='models/bert_fake_news_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load model if available
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"⚠️  Model not found at {model_path}")
            print("   Run 'python train_with_huggingface_dataset.py' first")
    
    def load_model(self):
        """Load the trained BERT model"""
        print(f"📁 Loading BERT model from {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            model_name = checkpoint.get('model_name', 'bert-base-uncased')
            self.model = BERTFakeNewsClassifier(model_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("✅ BERT model loaded successfully")
            
            # Display training stats if available
            if 'test_accuracies' in checkpoint:
                best_acc = max(checkpoint['test_accuracies'])
                print(f"📊 Best test accuracy: {best_acc:.1%}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def extract_text_from_url(self, url):
        """Extract text content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            for selector in ['h1', 'title', '[property="og:title"]']:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text().strip()
                    break
            
            # Extract content
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Combine title and content
            full_text = f"{title}. {content}"
            
            return {
                'title': title,
                'content': content[:1000],  # Limit content length
                'full_text': full_text[:2000],  # Limit for BERT
                'domain': urlparse(url).netloc.replace('www.', '')
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting text: {e}")
            return {
                'title': 'Could not extract title',
                'content': 'Could not extract content',
                'full_text': 'Could not extract text from URL',
                'domain': urlparse(url).netloc.replace('www.', '') if url else 'unknown'
            }
    
    def predict_text(self, text):
        """Predict if text is fake or real"""
        if self.model is None or self.tokenizer is None:
            return None
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'prediction': 'FAKE' if predicted_class == 1 else 'REAL',
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0][0].item(),
                'fake': probabilities[0][1].item()
            }
        }
    
    def predict(self, url):
        """Predict if news article at URL is fake or real"""
        print(f"🔍 Analyzing with BERT model: {url}")
        
        # Extract text from URL
        text_data = self.extract_text_from_url(url)
        
        if self.model is None:
            # Fallback to simple heuristics
            return self.fallback_prediction(url, text_data)
        
        # Use BERT model for prediction
        bert_result = self.predict_text(text_data['full_text'])
        
        if bert_result is None:
            return self.fallback_prediction(url, text_data)
        
        return {
            'url': url,
            'title': text_data['title'],
            'domain': text_data['domain'],
            'prediction': {
                'label': bert_result['prediction'],
                'confidence': bert_result['confidence'],
                'probabilities': bert_result['probabilities']
            },
            'method': 'BERT Neural Network',
            'model_type': 'bert-base-uncased'
        }
    
    def fallback_prediction(self, url, text_data):
        """Fallback prediction when BERT model is not available"""
        # Simple domain-based prediction
        trusted_domains = [
            'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com', 
            'washingtonpost.com', 'thehindu.com', 'timesofindia.indiatimes.com'
        ]
        
        domain = text_data['domain']
        is_trusted = any(trusted in domain for trusted in trusted_domains)
        
        if is_trusted:
            prediction = 'REAL'
            confidence = 0.85
        else:
            prediction = 'FAKE'
            confidence = 0.60
        
        return {
            'url': url,
            'title': text_data['title'],
            'domain': domain,
            'prediction': {
                'label': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': confidence if prediction == 'REAL' else 1 - confidence,
                    'fake': confidence if prediction == 'FAKE' else 1 - confidence
                }
            },
            'method': 'Fallback Domain-based',
            'model_type': 'rule-based'
        }

def main():
    parser = argparse.ArgumentParser(description='BERT Fake News Detector')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    parser.add_argument('--model', type=str, default='models/bert_fake_news_model.pth', help='Model path')
    
    args = parser.parse_args()
    
    print("🤖 BERT-Based Fake News Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = BERTFakeNewsDetector(args.model)
    
    # Analyze URL
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 30)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    print(f"🤖 Model: {result['model_type']}")
    
    prediction = result['prediction']
    emoji = "✅" if prediction['label'] == "REAL" else "❌"
    
    print(f"\n{emoji} PREDICTION: {prediction['label']}")
    print(f"🎯 Confidence: {prediction['confidence']:.1%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {prediction['probabilities']['real']:.1%}")
    print(f"   Fake: {prediction['probabilities']['fake']:.1%}")

if __name__ == '__main__':
    main()