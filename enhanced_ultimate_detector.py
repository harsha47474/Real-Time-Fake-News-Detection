#!/usr/bin/env python3

"""
Enhanced Ultimate Detector - Uses models trained on Indian news sources
Trained on: Hindustan Times, India Today, BBC News + Hugging Face Dataset
"""

import argparse
import pickle
import numpy as np
from huggingface_detector import HuggingFaceDetector
from domain_classifier import DomainClassifier
from consistent_detector import ConsistentFakeNewsDetector
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

class EnhancedRealNewsDetector:
    """Detector using models trained on real Indian news sources"""
    
    def __init__(self, model_path='models/enhanced_real_news_models.pkl'):
        self.model_path = model_path
        self.models = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load enhanced models trained on real news"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.models = data
                self.vectorizer = data['vectorizer']
            
            print("✅ Enhanced real news models loaded")
            accuracies = data['accuracies']
            print(f"📊 Enhanced LR Accuracy: {accuracies['logistic_regression']:.1%}")
            print(f"📊 Enhanced RF Accuracy: {accuracies['random_forest']:.1%}")
            
            # Show dataset info
            dataset_info = data.get('dataset_info', {})
            if dataset_info:
                print(f"📈 Trained on {dataset_info.get('total_samples', 0)} samples")
                sources = dataset_info.get('sources', {})
                print(f"📰 Sources: {', '.join(sources.keys())}")
            
        except Exception as e:
            print(f"❌ Error loading enhanced models: {e}")
            print("   Run 'python train_with_real_news.py' first")
            self.models = None
    
    def predict_with_model(self, text, model_name='logistic_regression'):
        """Predict using enhanced models"""
        if not self.models or not self.vectorizer:
            return None
        
        try:
            X = self.vectorizer.transform([text])
            model = self.models[model_name]
            
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
            print(f"⚠️  Enhanced model prediction error: {e}")
            return None
    
    def predict(self, url):
        """Predict using enhanced models"""
        print(f"🔍 Analyzing with enhanced real news models: {url}")
        
        # Extract text
        text_data = self.extract_text_from_url(url)
        if not text_data:
            return {'error': 'Could not extract content from URL'}
        
        text = text_data['full_text']
        
        if not self.models:
            return self.fallback_prediction(url, text_data)
        
        # Get predictions from both enhanced models
        lr_pred = self.predict_with_model(text, 'logistic_regression')
        rf_pred = self.predict_with_model(text, 'random_forest')
        
        if not lr_pred or not rf_pred:
            return self.fallback_prediction(url, text_data)
        
        # Enhanced ensemble (weighted by accuracy)
        lr_weight = self.models['accuracies']['logistic_regression']
        rf_weight = self.models['accuracies']['random_forest']
        total_weight = lr_weight + rf_weight
        
        final_real_prob = (
            lr_pred['probabilities']['real'] * lr_weight +
            rf_pred['probabilities']['real'] * rf_weight
        ) / total_weight
        
        final_fake_prob = 1 - final_real_prob
        
        if final_real_prob > final_fake_prob:
            final_label = 'REAL'
            confidence = final_real_prob
        else:
            final_label = 'FAKE'
            confidence = final_fake_prob
        
        return {
            'url': url,
            'title': text_data['title'],
            'domain': text_data['domain'],
            'prediction': {
                'label': final_label,
                'confidence': confidence,
                'probabilities': {
                    'real': final_real_prob,
                    'fake': final_fake_prob
                }
            },
            'individual_predictions': {
                'enhanced_lr': lr_pred,
                'enhanced_rf': rf_pred
            },
            'method': 'Enhanced Real News Models (Indian Sources)',
            'sources_trained_on': ['Hindustan Times', 'India Today', 'BBC News', 'Hugging Face Dataset']
        }
    
    def extract_text_from_url(self, url):
        """Extract text from URL"""
        if not url or not url.startswith(('http://', 'https://')):
            return None
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            full_text = f"{title_text}. {content}"
            
            return {
                'title': title_text,
                'content': content[:1000],
                'full_text': full_text,
                'domain': urlparse(url).netloc.replace('www.', '')
            }
            
        except Exception as e:
            print(f"⚠️  Error extracting text: {e}")
            return None
    
    def fallback_prediction(self, url, text_data):
        """Fallback prediction"""
        indian_sources = [
            'hindustantimes.com', 'indiatoday.in', 'timesofindia.indiatimes.com',
            'thehindu.com', 'ndtv.com', 'indianexpress.com'
        ]
        
        international_sources = [
            'bbc.com', 'reuters.com', 'cnn.com', 'nytimes.com'
        ]
        
        domain = text_data['domain']
        is_trusted = any(source in domain for source in indian_sources + international_sources)
        
        return {
            'url': url,
            'title': text_data['title'],
            'domain': domain,
            'prediction': {
                'label': 'REAL' if is_trusted else 'FAKE',
                'confidence': 0.8 if is_trusted else 0.6,
                'probabilities': {
                    'real': 0.8 if is_trusted else 0.4,
                    'fake': 0.2 if is_trusted else 0.6
                }
            },
            'method': 'Fallback Domain-based (Indian + International)',
            'sources_trained_on': []
        }

class SuperUltimateDetector:
    """Super Ultimate Detector combining ALL methods including enhanced real news models"""
    
    def __init__(self):
        print("🚀 Initializing Super Ultimate Detector...")
        print("🔄 Loading ALL available detectors including enhanced models...")
        
        # Initialize all detectors
        self.hf_detector = HuggingFaceDetector()
        self.enhanced_detector = EnhancedRealNewsDetector()
        self.domain_classifier = DomainClassifier()
        self.consistent_detector = ConsistentFakeNewsDetector()
        
        # Check availability
        self.has_hf_models = (self.hf_detector.models is not None)
        self.has_enhanced_models = (self.enhanced_detector.models is not None)
        
        # Enhanced keyword detection for Indian context
        self.fake_keywords = [
            'fake', 'hoax', 'scam', 'conspiracy', 'debunked', 'false', 'misleading',
            'clickbait', 'satire', 'parody', 'rumor', 'unverified', 'alleged',
            'breaking', 'shocking', 'unbelievable', 'exclusive', 'leaked',
            'secret', 'hidden truth', 'they dont want you to know',
            'doctors hate this', 'miracle cure', 'government cover-up',
            # Indian context keywords
            'anti-national', 'urban naxal', 'tukde tukde', 'fake encounter',
            'paid media', 'presstitute', 'godi media'
        ]
        
        print(f"✅ Hugging Face Models (97%+): {'Available' if self.has_hf_models else 'Not Available'}")
        print(f"✅ Enhanced Real News Models: {'Available' if self.has_enhanced_models else 'Not Available'}")
        print("✅ Domain Classifier: Available")
        print("✅ Rule-based Detector: Available")
        print("✅ Enhanced Keyword Detection: Available")
        
        total_methods = sum([
            self.has_hf_models,
            self.has_enhanced_models,
            True,  # Domain classifier
            True,  # Rule-based
            True   # Keyword detection
        ])
        print(f"🔥 Total methods available: {total_methods}")
        
        if self.has_enhanced_models:
            print("🎯 Using enhanced models trained on Indian news sources!")
    
    def predict(self, url):
        """Super ultimate prediction using all methods"""
        print(f"🔍 Super ultimate analysis of: {url}")
        
        results = {}
        
        # Enhanced real news models (highest priority for Indian sources)
        if self.has_enhanced_models:
            try:
                enhanced_result = self.enhanced_detector.predict(url)
                if 'error' not in enhanced_result:
                    results['enhanced'] = enhanced_result
            except Exception as e:
                print(f"⚠️  Enhanced models failed: {e}")
                results['enhanced'] = None
        
        # Hugging Face models
        if self.has_hf_models:
            try:
                hf_result = self.hf_detector.predict(url)
                if 'error' not in hf_result:
                    results['huggingface'] = hf_result
            except Exception as e:
                print(f"⚠️  Hugging Face models failed: {e}")
                results['huggingface'] = None
        
        # Domain classifier
        try:
            domain_result = self.domain_classifier.classify(url)
            results['domain'] = domain_result
        except Exception as e:
            print(f"⚠️  Domain classification failed: {e}")
            results['domain'] = None
        
        # Rule-based detector
        try:
            consistent_result = self.consistent_detector.predict(url)
            results['consistent'] = consistent_result
        except Exception as e:
            print(f"⚠️  Consistent prediction failed: {e}")
            results['consistent'] = None
        
        return self.super_ensemble(url, results)
    
    def super_ensemble(self, url, results):
        """Super ensemble with enhanced weighting"""
        predictions = []
        weights = []
        methods = []
        
        # Enhanced models (highest weight for Indian sources)
        if results.get('enhanced'):
            enhanced_pred = results['enhanced']['prediction']
            enhanced_real_prob = enhanced_pred['probabilities']['real']
            confidence = enhanced_pred['confidence']
            
            # Very high weight for enhanced models
            weight = 0.5 if confidence > 0.8 else 0.4
            
            predictions.append(enhanced_real_prob)
            weights.append(weight)
            methods.append(f"Enhanced Models ({confidence:.1%})")
        
        # Hugging Face models
        if results.get('huggingface'):
            hf_pred = results['huggingface']['prediction']
            hf_real_prob = hf_pred['probabilities']['real']
            confidence = hf_pred['confidence']
            
            weight = 0.3 if confidence > 0.8 else 0.25
            
            predictions.append(hf_real_prob)
            weights.append(weight)
            methods.append(f"HF Models ({confidence:.1%})")
        
        # Domain classifier
        if results.get('domain'):
            domain_pred = results['domain']
            trust_score = domain_pred['trust_score']
            
            if domain_pred['prediction'] == 'REAL':
                domain_real_prob = domain_pred['confidence']
            else:
                domain_real_prob = 1 - domain_pred['confidence']
            
            weight = 0.2 if trust_score > 0.8 else 0.15
            
            predictions.append(domain_real_prob)
            weights.append(weight)
            methods.append(f"Domain ({domain_pred['confidence']:.1%})")
        
        # Rule-based detector
        if results.get('consistent'):
            consistent_pred = results['consistent']['prediction']
            consistent_real_prob = consistent_pred['probabilities']['real']
            
            predictions.append(consistent_real_prob)
            weights.append(0.1)
            methods.append(f"Rules ({consistent_pred['confidence']:.1%})")
        
        # Calculate final prediction
        if predictions:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            final_real_prob = sum(p * w for p, w in zip(predictions, normalized_weights))
            final_fake_prob = 1 - final_real_prob
            
            if final_real_prob > final_fake_prob:
                final_label = 'REAL'
                confidence = final_real_prob
            else:
                final_label = 'FAKE'
                confidence = final_fake_prob
        else:
            final_label = 'UNKNOWN'
            confidence = 0.5
            final_real_prob = 0.5
            final_fake_prob = 0.5
        
        # Get basic info
        title = "Unknown"
        domain = "unknown"
        
        for result_key in ['enhanced', 'huggingface', 'domain', 'consistent']:
            if results.get(result_key):
                if 'title' in results[result_key]:
                    title = results[result_key]['title']
                if 'domain' in results[result_key]:
                    domain = results[result_key]['domain']
                break
        
        return {
            'url': url,
            'title': title,
            'domain': domain,
            'final_prediction': {
                'label': final_label,
                'confidence': confidence,
                'probabilities': {
                    'real': final_real_prob,
                    'fake': final_fake_prob
                }
            },
            'individual_predictions': {
                'enhanced': results.get('enhanced', {}).get('prediction'),
                'huggingface': results.get('huggingface', {}).get('prediction'),
                'domain': {
                    'prediction': results['domain']['prediction'],
                    'confidence': results['domain']['confidence'],
                    'trust_score': results['domain']['trust_score']
                } if results.get('domain') else None,
                'consistent': results.get('consistent', {}).get('prediction')
            },
            'ensemble_info': {
                'methods_used': methods,
                'weights': dict(zip(['Enhanced', 'HF', 'Domain', 'Rules'], 
                               normalized_weights[:len(methods)])) if predictions else {},
                'num_methods': len(predictions),
                'consensus': {
                    'real_votes': sum(1 for p in predictions if p > 0.5),
                    'fake_votes': sum(1 for p in predictions if p < 0.5)
                } if predictions else {'real_votes': 0, 'fake_votes': 0}
            },
            'method': f'Super Ultimate Ensemble ({len(predictions)} methods: Enhanced Indian + HF + Domain + Rules)',
            'trained_on': ['Hindustan Times', 'India Today', 'BBC News', 'Hugging Face Dataset']
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced Ultimate Fake News Detector')
    parser.add_argument('--url', type=str, help='URL to analyze')
    parser.add_argument('--train', action='store_true', help='Train enhanced models first')
    
    args = parser.parse_args()
    
    if args.train:
        print("🏋️ Training enhanced models first...")
        os.system("python train_with_real_news.py")
        return
    
    if not args.url:
        print("❌ Please provide a URL to analyze or use --train to train models")
        print("💡 Examples:")
        print("   python enhanced_ultimate_detector.py --url \"https://www.hindustantimes.com/india-news\"")
        print("   python enhanced_ultimate_detector.py --train")
        return
    
    print("🏆 Enhanced Ultimate Fake News Detector")
    print("=" * 70)
    print("🎯 Enhanced with Indian News Sources: Hindustan Times, India Today, BBC")
    
    detector = SuperUltimateDetector()
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 ENHANCED ANALYSIS RESULTS")
    print("=" * 40)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    final = result['final_prediction']
    emoji = "✅" if final['label'] == "REAL" else "❌"
    print(f"\n{emoji} FINAL PREDICTION: {final['label']}")
    print(f"🎯 Confidence: {final['confidence']:.1%}")
    
    # Show training sources
    if 'trained_on' in result:
        print(f"📚 Trained on: {', '.join(result['trained_on'])}")
    
    print(f"\n🏆 Enhanced analysis completed!")

if __name__ == '__main__':
    main()