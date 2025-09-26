#!/usr/bin/env python3

"""
Enhanced Fake News Detector combining BERT + Domain Analysis + Content Analysis
"""

import argparse
from bert_detector import BERTFakeNewsDetector
from domain_classifier import DomainClassifier
from consistent_detector import ConsistentFakeNewsDetector
import numpy as np

class EnhancedFakeNewsDetector:
    """Enhanced detector combining multiple approaches"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Fake News Detector...")
        
        # Initialize all detectors
        self.bert_detector = BERTFakeNewsDetector()
        self.domain_classifier = DomainClassifier()
        self.consistent_detector = ConsistentFakeNewsDetector()
        
        # Check which models are available
        self.has_bert = self.bert_detector.model is not None
        
        print(f"✅ BERT Model: {'Available' if self.has_bert else 'Not Available (using fallback)'}")
        print("✅ Domain Classifier: Available")
        print("✅ Consistent Detector: Available")
    
    def predict(self, url):
        """Enhanced prediction using multiple methods"""
        print(f"🔍 Enhanced analysis of: {url}")
        
        results = {}
        
        # Get predictions from all available methods
        try:
            bert_result = self.bert_detector.predict(url)
            results['bert'] = bert_result
        except Exception as e:
            print(f"⚠️  BERT prediction failed: {e}")
            results['bert'] = None
        
        try:
            domain_result = self.domain_classifier.classify(url)
            results['domain'] = domain_result
        except Exception as e:
            print(f"⚠️  Domain classification failed: {e}")
            results['domain'] = None
        
        try:
            consistent_result = self.consistent_detector.predict(url)
            results['consistent'] = consistent_result
        except Exception as e:
            print(f"⚠️  Consistent prediction failed: {e}")
            results['consistent'] = None
        
        # Combine predictions using weighted ensemble
        return self.ensemble_prediction(url, results)
    
    def ensemble_prediction(self, url, results):
        """Combine predictions from multiple models"""
        predictions = []
        weights = []
        methods = []
        
        # BERT prediction (highest weight if available)
        if results['bert'] and self.has_bert:
            bert_pred = results['bert']['prediction']
            bert_real_prob = bert_pred['probabilities']['real']
            predictions.append(bert_real_prob)
            weights.append(0.5)  # 50% weight for BERT
            methods.append(f"BERT ({bert_pred['confidence']:.1%})")
        
        # Domain classifier (reliable for known domains)
        if results['domain']:
            domain_pred = results['domain']
            if domain_pred['prediction'] == 'REAL':
                domain_real_prob = domain_pred['confidence']
            else:
                domain_real_prob = 1 - domain_pred['confidence']
            predictions.append(domain_real_prob)
            weights.append(0.3)  # 30% weight for domain
            methods.append(f"Domain ({domain_pred['confidence']:.1%})")
        
        # Consistent detector (rule-based backup)
        if results['consistent']:
            consistent_pred = results['consistent']['prediction']
            consistent_real_prob = consistent_pred['probabilities']['real']
            predictions.append(consistent_real_prob)
            weights.append(0.2)  # 20% weight for consistent
            methods.append(f"Rules ({consistent_pred['confidence']:.1%})")
        
        # Calculate weighted average
        if predictions:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average
            final_real_prob = sum(p * w for p, w in zip(predictions, normalized_weights))
            final_fake_prob = 1 - final_real_prob
            
            # Final prediction
            if final_real_prob > final_fake_prob:
                final_label = 'REAL'
                confidence = final_real_prob
            else:
                final_label = 'FAKE'
                confidence = final_fake_prob
        else:
            # Fallback if all methods failed
            final_label = 'UNKNOWN'
            confidence = 0.5
            final_real_prob = 0.5
            final_fake_prob = 0.5
        
        # Get basic info from any available result
        title = "Unknown"
        domain = "unknown"
        
        for result_key in ['bert', 'domain', 'consistent']:
            if results[result_key]:
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
                'bert': results['bert']['prediction'] if results['bert'] else None,
                'domain': {
                    'prediction': results['domain']['prediction'],
                    'confidence': results['domain']['confidence'],
                    'trust_score': results['domain']['trust_score']
                } if results['domain'] else None,
                'consistent': results['consistent']['prediction'] if results['consistent'] else None
            },
            'ensemble_info': {
                'methods_used': methods,
                'weights': dict(zip(['BERT', 'Domain', 'Rules'], normalized_weights)) if predictions else {},
                'num_methods': len(predictions)
            },
            'method': 'Enhanced Ensemble (BERT + Domain + Rules)'
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced Fake News Detector')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Fake News Detector")
    print("=" * 60)
    print("🤖 Combining BERT + Domain Analysis + Rule-based Methods")
    
    # Initialize detector
    detector = EnhancedFakeNewsDetector()
    
    # Analyze URL
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 ENHANCED ANALYSIS RESULTS")
    print("=" * 40)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    # Final prediction
    final = result['final_prediction']
    emoji = "✅" if final['label'] == "REAL" else "❌" if final['label'] == "FAKE" else "❓"
    print(f"\n{emoji} FINAL PREDICTION: {final['label']}")
    print(f"🎯 Confidence: {final['confidence']:.1%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {final['probabilities']['real']:.1%}")
    print(f"   Fake: {final['probabilities']['fake']:.1%}")
    
    # Ensemble info
    ensemble = result['ensemble_info']
    print(f"\n🔬 ENSEMBLE DETAILS:")
    print(f"📈 Methods used: {ensemble['num_methods']}")
    for method in ensemble['methods_used']:
        print(f"   • {method}")
    
    # Individual predictions
    individual = result['individual_predictions']
    print(f"\n🔍 INDIVIDUAL PREDICTIONS:")
    
    if individual['bert']:
        bert = individual['bert']
        print(f"🤖 BERT: {bert['label']} ({bert['confidence']:.1%})")
    
    if individual['domain']:
        domain = individual['domain']
        print(f"🌐 Domain: {domain['prediction']} ({domain['confidence']:.1%})")
    
    if individual['consistent']:
        consistent = individual['consistent']
        print(f"📋 Rules: {consistent['label']} ({consistent['confidence']:.1%})")
    
    print(f"\n💡 This enhanced detector provides the most accurate results!")

if __name__ == '__main__':
    main()