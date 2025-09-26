#!/usr/bin/env python3

"""
Comprehensive test of the ultimate detector
"""

from ultimate_detector import UltimateDetector

def test_comprehensive():
    print("🧪 Comprehensive Ultimate Detector Test")
    print("=" * 60)
    
    detector = UltimateDetector()
    
    # Test different types of URLs
    test_urls = [
        {
            'url': 'https://www.reuters.com/world/',
            'expected': 'REAL',
            'description': 'Trusted news source (Reuters)'
        },
        {
            'url': 'https://www.bbc.com/news',
            'expected': 'REAL', 
            'description': 'Trusted news source (BBC)'
        },
        {
            'url': 'https://timesofindia.indiatimes.com/business/india-business/trumps-100-tariffs-on-pharma-sun-pharma-biocon-cipla-other-pharmaceutical-stocks-tank-jitters-on-d-street/articleshow/124144816.cms',
            'expected': 'REAL',
            'description': 'Times of India business article'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_urls, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"URL: {test_case['url']}")
        print(f"Expected: {test_case['expected']}")
        print("-" * 50)
        
        try:
            result = detector.predict(test_case['url'])
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                results.append({'test': i, 'status': 'ERROR', 'error': result['error']})
                continue
            
            prediction = result['final_prediction']['label']
            confidence = result['final_prediction']['confidence']
            
            # Check if prediction matches expected
            is_correct = prediction == test_case['expected']
            status_emoji = "✅" if is_correct else "❌"
            
            print(f"{status_emoji} Prediction: {prediction} ({confidence:.1%})")
            print(f"📰 Title: {result['title']}")
            print(f"🌐 Domain: {result['domain']}")
            print(f"🔍 Method: {result['method']}")
            
            # Show ensemble details
            ensemble = result['ensemble_info']
            print(f"📊 Methods used: {ensemble['num_methods']}")
            print(f"🗳️  Voting: {ensemble['consensus']['real_votes']} Real, {ensemble['consensus']['fake_votes']} Fake")
            
            # Show individual predictions
            individual = result['individual_predictions']
            print("🔍 Individual results:")
            for method, pred in individual.items():
                if isinstance(pred, dict) and 'prediction' in pred:
                    print(f"   • {method}: {pred['prediction']} ({pred.get('confidence', 0):.1%})")
                elif isinstance(pred, dict) and 'label' in pred:
                    print(f"   • {method}: {pred['label']} ({pred.get('confidence', 0):.1%})")
            
            # Show detected keywords if any
            if result.get('detected_keywords'):
                print(f"🚨 Keywords: {', '.join(result['detected_keywords'][:3])}")
            
            results.append({
                'test': i,
                'status': 'CORRECT' if is_correct else 'INCORRECT',
                'prediction': prediction,
                'expected': test_case['expected'],
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({'test': i, 'status': 'EXCEPTION', 'error': str(e)})
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    
    correct = sum(1 for r in results if r['status'] == 'CORRECT')
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"✅ Correct predictions: {correct}/{total} ({accuracy:.1f}%)")
    
    for result in results:
        status_emoji = "✅" if result['status'] == 'CORRECT' else "❌"
        print(f"{status_emoji} Test {result['test']}: {result['status']}")
        if 'prediction' in result:
            print(f"   Predicted: {result['prediction']} (Expected: {result['expected']})")

if __name__ == '__main__':
    test_comprehensive()