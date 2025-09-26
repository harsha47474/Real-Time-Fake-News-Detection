#!/usr/bin/env python3

"""
Test the fast-trained models
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import argparse
from fast_train_cpu import FastTextClassifier
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class FastModelDetector:
    """Detector using fast-trained models"""
    
    def __init__(self):
        self.sklearn_models = None
        self.nn_model = None
        self.vectorizer = None
        self.word_to_idx = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load sklearn models
            with open('models/fast_models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.sklearn_models = data
                self.vectorizer = data['vectorizer']
            print("✅ Sklearn models loaded")
            
            # Load neural network
            nn_data = torch.load('models/fast_neural_model.pth', map_location='cpu')
            self.word_to_idx = nn_data['word_to_idx']
            vocab_size = nn_data['vocab_size']
            max_length = nn_data['max_length']
            
            self.nn_model = FastTextClassifier(vocab_size, embedding_dim=50, hidden_dim=64)
            self.nn_model.load_state_dict(nn_data['model_state_dict'])
            self.nn_model.eval()
            self.max_length = max_length
            print("✅ Neural network loaded")
            
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
            print("   Run 'python fast_train_cpu.py' first")
    
    def extract_text_from_url(self, url):
        """Extract text from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
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
            
            full_text = f"{title}. {content}"[:1000]  # Limit length
            
            return {
                'title': title,
                'content': content[:500],
                'full_text': full_text,
                'domain': urlparse(url).netloc.replace('www.', '')
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting text: {e}")
            return {
                'title': 'Could not extract title',
                'content': 'Could not extract content',
                'full_text': 'Could not extract text',
                'domain': urlparse(url).netloc.replace('www.', '') if url else 'unknown'
            }
    
    def predict_with_sklearn(self, text, model_name='logistic_regression'):
        """Predict using sklearn models"""
        if not self.sklearn_models or not self.vectorizer:
            return None
        
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Get model
            model = self.sklearn_models[model_name]
            
            # Predict
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            return {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': max(probabilities),
                'probabilities': {
                    'real': probabilities[0],
                    'fake': probabilities[1]
                }
            }
            
        except Exception as e:
            print(f"⚠️  Sklearn prediction error: {e}")
            return None
    
    def predict_with_nn(self, text):
        """Predict using neural network"""
        if not self.nn_model or not self.word_to_idx:
            return None
        
        try:
            # Tokenize and convert to indices
            tokens = text.lower().split()
            indices = [self.word_to_idx.get(token, 0) for token in tokens[:self.max_length]]
            indices += [0] * (self.max_length - len(indices))
            
            # Convert to tensor
            X = torch.tensor([indices], dtype=torch.long)
            
            # Predict
            with torch.no_grad():
                outputs = self.nn_model(X)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            return {
                'prediction': 'FAKE' if predicted_class == 1 else 'REAL',
                'confidence': probabilities[predicted_class].item(),
                'probabilities': {
                    'real': probabilities[0].item(),
                    'fake': probabilities[1].item()
                }
            }
            
        except Exception as e:
            print(f"⚠️  Neural network prediction error: {e}")
            return None
    
    def predict(self, url):
        """Predict using ensemble of fast models"""
        print(f"🔍 Analyzing with fast models: {url}")
        
        # Extract text
        text_data = self.extract_text_from_url(url)
        text = text_data['full_text']
        
        # Get predictions from all models
        predictions = {}
        
        # Logistic Regression
        lr_pred = self.predict_with_sklearn(text, 'logistic_regression')
        if lr_pred:
            predictions['logistic_regression'] = lr_pred
        
        # Random Forest
        rf_pred = self.predict_with_sklearn(text, 'random_forest')
        if rf_pred:
            predictions['random_forest'] = rf_pred
        
        # Neural Network
        nn_pred = self.predict_with_nn(text)
        if nn_pred:
            predictions['neural_network'] = nn_pred
        
        # Ensemble prediction
        if predictions:
            # Average probabilities
            real_probs = [pred['probabilities']['real'] for pred in predictions.values()]
            fake_probs = [pred['probabilities']['fake'] for pred in predictions.values()]
            
            avg_real_prob = np.mean(real_probs)
            avg_fake_prob = np.mean(fake_probs)
            
            final_prediction = 'REAL' if avg_real_prob > avg_fake_prob else 'FAKE'
            confidence = max(avg_real_prob, avg_fake_prob)
            
        else:
            # Fallback
            final_prediction = 'REAL'
            confidence = 0.5
            avg_real_prob = 0.5
            avg_fake_prob = 0.5
        
        return {
            'url': url,
            'title': text_data['title'],
            'domain': text_data['domain'],
            'prediction': {
                'label': final_prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': avg_real_prob,
                    'fake': avg_fake_prob
                }
            },
            'individual_predictions': predictions,
            'method': 'Fast Ensemble (LR + RF + NN)'
        }

def main():
    parser = argparse.ArgumentParser(description='Test Fast Models')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    
    print("⚡ Fast Model Fake News Detector")
    print("=" * 50)
    
    # Initialize detector
    detector = FastModelDetector()
    
    # Test prediction
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 FAST MODEL RESULTS")
    print("=" * 30)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    prediction = result['prediction']
    emoji = "✅" if prediction['label'] == "REAL" else "❌"
    
    print(f"\n{emoji} PREDICTION: {prediction['label']}")
    print(f"🎯 Confidence: {prediction['confidence']:.1%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {prediction['probabilities']['real']:.1%}")
    print(f"   Fake: {prediction['probabilities']['fake']:.1%}")
    
    # Individual model results
    individual = result['individual_predictions']
    print(f"\n🔍 INDIVIDUAL MODEL RESULTS:")
    
    for model_name, pred in individual.items():
        print(f"   {model_name}: {pred['prediction']} ({pred['confidence']:.1%})")

if __name__ == '__main__':
    main()