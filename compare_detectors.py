#!/usr/bin/env python3

"""
Compare different detectors to show consistency vs inconsistency
"""

import argparse
from domain_classifier import DomainClassifier
from consistent_detector import ConsistentFakeNewsDetector
from fixed_improved_detector import FixedImprovedDetector

def test_consistency(url, detector_name, detector, runs=5):
    """Test detector consistency"""
    print(f"\n🧪 Testing {detector_name} ({runs} runs):")
    results = []
    
    for i in range(runs):
        if hasattr(detector, 'predict'):
            result = detector.predict(url)
            if 'prediction' in result:
                # Handle different result formats
                if isinstance(result['prediction'], dict):
                    label = result['prediction']['label']
                    confidence = result['prediction']['confidence']
                else:
                    label = result['prediction']
                    confidence = result['confidence']
            elif 'final_prediction' in result:
                label = result['final_prediction']['label']
                confidence = result['final_prediction']['confidence']
            else:
                label = "Unknown"
                confidence = 0.5
        else:
            result = detector.classify(url)
            label = result['prediction']
            confidence = result['confidence']
        
        results.append((label, confidence))
        print(f"   Run {i+1}: {label} ({confidence:.1%})")
    
    # Check consistency
    labels = [r[0] for r in results]
    confidences = [r[1] for r in results]
    
    is_consistent = len(set(labels)) == 1 and len(set(confidences)) == 1
    
    if is_consistent:
        print(f"   ✅ CONSISTENT - Same result every time!")
    else:
        print(f"   ❌ INCONSISTENT - Results vary!")
        print(f"   Labels: {set(labels)}")
        print(f"   Confidence range: {min(confidences):.1%} - {max(confidences):.1%}")
    
    return is_consistent

def main():
    parser = argparse.ArgumentParser(description='Compare Detector Consistency')
    parser.add_argument('--url', type=str, required=True, help='URL to test')
    
    args = parser.parse_args()
    
    print("🔬 Detector Consistency Comparison")
    print("=" * 60)
    print(f"Testing URL: {args.url}")
    
    # Test different detectors
    detectors = [
        ("Domain Classifier", DomainClassifier()),
        ("Consistent Detector", ConsistentFakeNewsDetector()),
        ("Fixed Improved Detector", FixedImprovedDetector())
    ]
    
    consistent_count = 0
    
    for name, detector in detectors:
        try:
            is_consistent = test_consistency(args.url, name, detector)
            if is_consistent:
                consistent_count += 1
        except Exception as e:
            print(f"   ❌ Error testing {name}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Consistent detectors: {consistent_count}/{len(detectors)}")
    print(f"❌ Inconsistent detectors: {len(detectors) - consistent_count}/{len(detectors)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("✅ Use: Domain Classifier, Consistent Detector, or Fixed Improved Detector")
    print("❌ Avoid: Original GNN models with random components")
    print("🎯 Best choice: Fixed Improved Detector (combines multiple approaches)")

if __name__ == '__main__':
    main()