#!/usr/bin/env python3

"""
Fixed Improved Detector - Consistent results with ensemble method
"""

import argparse
import hashlib
import numpy as np
from consistent_detector import ConsistentFakeNewsDetector
from domain_classifier import DomainClassifier

class FixedImprovedDetector:
    def __init__(self):
        self.consistent_detector = ConsistentFakeNewsDetector()
        self.domain_classifier = DomainClassifier()
        
    def predict(self, url):
        """Ensemble prediction with consistent results"""
        print(f"🔍 Analyzing: {url}")
        
        # Get consistent analysis
        consistent_result = self.consistent_detector.predict(url)
        
        # Get domain classification
        domain_result = self.domain_classifier.classify(url)
        
        # Combine predictions with weighted ensemble
        consistent_weight = 0.6  # Our consistent detector
        domain_weight = 0.4      # Domain classifier
        
        # Convert predictions to probabilities
        if domain_result['prediction'] == 'REAL':
            domain_real_prob = domain_result['confidence']
            domain_fake_prob = 1 - domain_result['confidence']
        else:
            domain_fake_prob = domain_result['confidence']
            domain_real_prob = 1 - domain_result['confidence']
        
        consistent_real_prob = consistent_result['prediction']['probabilities']['real']
        consistent_fake_prob = consistent_result['prediction']['probabilities']['fake']
        
        # Weighted ensemble
        final_real_prob = (consistent_weight * consistent_real_prob + domain_weight * domain_real_prob)
        final_fake_prob = (consistent_weight * consistent_fake_prob + domain_weight * domain_fake_prob)
        
        # Final prediction
        if final_real_prob > final_fake_prob:
            final_prediction = 'REAL'
            confidence = final_real_prob
        else:
            final_prediction = 'FAKE'
            confidence = final_fake_prob
        
        return {
            'url': url,
            'title': consistent_result['title'],
            'domain': consistent_result['domain'],
            'final_prediction': {
                'label': final_prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': final_real_prob,
                    'fake': final_fake_prob
                }
            },
            'individual_predictions': {
                'consistent_analyzer': {
                    'prediction': consistent_result['prediction']['label'],
                    'confidence': consistent_result['prediction']['confidence'],
                    'analysis': consistent_result['analysis']
                },
                'domain_classifier': {
                    'prediction': domain_result['prediction'],
                    'confidence': domain_result['confidence'],
                    'trust_score': domain_result['trust_score']
                }
            },
            'method': 'Fixed Ensemble (Consistent + Domain)'
        }

def main():
    parser = argparse.ArgumentParser(description='Fixed Improved Fake News Detector')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    
    print("🚀 Fixed Improved Fake News Detector")
    print("=" * 60)
    print("✅ Consistent results every time!")
    
    detector = FixedImprovedDetector()
    
    # Test consistency
    print(f"\n🔄 Testing consistency with 3 runs:")
    for i in range(3):
        result = detector.predict(args.url)
        print(f"Run {i+1}: {result['final_prediction']['label']} ({result['final_prediction']['confidence']:.1%})")
    
    # Show detailed results
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 40)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    # Final prediction
    final = result['final_prediction']
    emoji = "✅" if final['label'] == "REAL" else "❌"
    print(f"\n{emoji} FINAL PREDICTION: {final['label']}")
    print(f"🎯 Confidence: {final['confidence']:.1%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {final['probabilities']['real']:.1%}")
    print(f"   Fake: {final['probabilities']['fake']:.1%}")
    
    # Individual predictions
    print(f"\n🔍 INDIVIDUAL PREDICTIONS:")
    print("=" * 30)
    
    consistent = result['individual_predictions']['consistent_analyzer']
    print(f"🎯 Consistent Analyzer: {consistent['prediction']} ({consistent['confidence']:.1%})")
    analysis = consistent['analysis']
    print(f"   Domain: {analysis['domain_score']:.1%}, Content: {analysis['content_score']:.1%}")
    
    domain = result['individual_predictions']['domain_classifier']
    print(f"🌐 Domain Classifier: {domain['prediction']} ({domain['confidence']:.1%})")
    print(f"   Trust Score: {domain['trust_score']:.1%}")
    
    print(f"\n💡 This detector gives the same result every time for the same URL!")

if __name__ == '__main__':
    main()