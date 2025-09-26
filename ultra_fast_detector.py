#!/usr/bin/env python3

"""
Ultra Fast Fake News Detector - Combines all approaches for maximum accuracy
"""

import argparse
from test_fast_models import FastModelDetector
from domain_classifier import DomainClassifier
from consistent_detector import ConsistentFakeNewsDetector
import numpy as np

class UltraFastDetector:
    """Ultra-fast detector combining all available methods"""
    
    def __init__(self):
        print("🚀 Initializing Ultra Fast Detector...")
        
        # Initialize all detectors
        self.fast_models = FastModelDetector()
        self.domain_classifier = DomainClassifier()
        self.consistent_detector = ConsistentFakeNewsDetector()
        
        # Check availability
        self.has_fast_models = (self.fast_models.sklearn_models is not None)
        
        print(f"✅ Fast ML Models: {'Available' if self.has_fast_models else 'Not Available'}")
        print("✅ Domain Classifier: Available")
        print("✅ Consistent Detector: Available")
    
    def predict(self, url):
        """Ultra-fast prediction using all available methods"""
        print(f"⚡ Ultra-fast analysis of: {url}")
        
        results = {}
        
        # Get predictions from all methods
        try:
            if self.has_fast_models:
                fast_result = self.fast_models.predict(url)
                results['fast_models'] = fast_result
        except Exception as e:
            print(f"⚠️  Fast models failed: {e}")
            results['fast_models'] = None
        
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
        
        # Combine all predictions
        return self.ultra_ensemble(url, results)
    
    def ultra_ensemble(self, url, results):
        """Ultra ensemble combining all methods with smart weighting"""
        predictions = []
        weights = []
        methods = []
        
        # Fast ML models (highest weight if available and confident)
        if results['fast_models'] and self.has_fast_models:
            fast_pred = results['fast_models']['prediction']
            fast_real_prob = fast_pred['probabilities']['real']
            
            # Weight based on confidence
            confidence = fast_pred['confidence']
            if confidence > 0.7:
                weight = 0.5  # High confidence
            elif confidence > 0.6:
                weight = 0.4  # Medium confidence
            else:
                weight = 0.3  # Low confidence
            
            predictions.append(fast_real_prob)
            weights.append(weight)
            methods.append(f"ML Models ({confidence:.1%})")
        
        # Domain classifier (very reliable for known domains)
        if results['domain']:
            domain_pred = results['domain']
            trust_score = domain_pred['trust_score']
            
            if domain_pred['prediction'] == 'REAL':
                domain_real_prob = domain_pred['confidence']
            else:
                domain_real_prob = 1 - domain_pred['confidence']
            
            # Higher weight for high trust domains
            if trust_score > 0.8:
                weight = 0.4  # Very trusted domain
            elif trust_score > 0.6:
                weight = 0.3  # Somewhat trusted
            else:
                weight = 0.2  # Unknown domain
            
            predictions.append(domain_real_prob)
            weights.append(weight)
            methods.append(f"Domain ({domain_pred['confidence']:.1%})")
        
        # Consistent detector (rule-based backup)
        if results['consistent']:
            consistent_pred = results['consistent']['prediction']
            consistent_real_prob = consistent_pred['probabilities']['real']
            
            predictions.append(consistent_real_prob)
            weights.append(0.2)  # Lower weight for rule-based
            methods.append(f"Rules ({consistent_pred['confidence']:.1%})")
        
        # Calculate weighted ensemble
        if predictions:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average
            final_real_prob = sum(p * w for p, w in zip(predictions, normalized_weights))
            final_fake_prob = 1 - final_real_prob
            
            # Apply confidence boost for unanimous decisions
            unanimous_real = all(p > 0.5 for p in predictions)
            unanimous_fake = all(p < 0.5 for p in predictions)
            
            if unanimous_real or unanimous_fake:
                # Boost confidence for unanimous decisions
                if final_real_prob > 0.5:
                    final_real_prob = min(0.95, final_real_prob + 0.1)
                else:
                    final_real_prob = max(0.05, final_real_prob - 0.1)
                final_fake_prob = 1 - final_real_prob
            
            # Final prediction
            if final_real_prob > final_fake_prob:
                final_label = 'REAL'
                confidence = final_real_prob
            else:
                final_label = 'FAKE'
                confidence = final_fake_prob
        else:
            # Fallback
            final_label = 'UNKNOWN'
            confidence = 0.5
            final_real_prob = 0.5
            final_fake_prob = 0.5
        
        # Get basic info
        title = "Unknown"
        domain = "unknown"
        
        for result_key in ['fast_models', 'domain', 'consistent']:
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
                'fast_models': results['fast_models']['prediction'] if results['fast_models'] else None,
                'domain': {
                    'prediction': results['domain']['prediction'],
                    'confidence': results['domain']['confidence'],
                    'trust_score': results['domain']['trust_score']
                } if results['domain'] else None,
                'consistent': results['consistent']['prediction'] if results['consistent'] else None
            },
            'ensemble_info': {
                'methods_used': methods,
                'weights': dict(zip(['ML', 'Domain', 'Rules'], normalized_weights)) if predictions else {},
                'num_methods': len(predictions),
                'unanimous': len(set(p > 0.5 for p in predictions)) == 1 if predictions else False
            },
            'method': 'Ultra Fast Ensemble (ML + Domain + Rules)'
        }

def main():
    parser = argparse.ArgumentParser(description='Ultra Fast Fake News Detector')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    
    print("⚡ Ultra Fast Fake News Detector")
    print("=" * 60)
    print("🚀 Lightning-fast analysis with maximum accuracy!")
    
    # Initialize detector
    detector = UltraFastDetector()
    
    # Analyze URL
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📊 ULTRA FAST ANALYSIS RESULTS")
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
    print(f"🤝 Unanimous decision: {'Yes' if ensemble['unanimous'] else 'No'}")
    for method in ensemble['methods_used']:
        print(f"   • {method}")
    
    # Individual predictions
    individual = result['individual_predictions']
    print(f"\n🔍 INDIVIDUAL PREDICTIONS:")
    
    if individual['fast_models']:
        fast = individual['fast_models']
        print(f"🤖 ML Models: {fast['label']} ({fast['confidence']:.1%})")
    
    if individual['domain']:
        domain = individual['domain']
        print(f"🌐 Domain: {domain['prediction']} ({domain['confidence']:.1%})")
    
    if individual['consistent']:
        consistent = individual['consistent']
        print(f"📋 Rules: {consistent['label']} ({consistent['confidence']:.1%})")
    
    print(f"\n⚡ Ultra-fast analysis completed!")

if __name__ == '__main__':
    main()