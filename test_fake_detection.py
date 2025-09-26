#!/usr/bin/env python3

"""
Test the fake keyword detection
"""

from ultimate_detector import UltimateDetector

def test_fake_detection():
    print("🧪 Testing Fake Keyword Detection")
    print("=" * 50)
    
    detector = UltimateDetector()
    
    # Test URLs with fake content
    test_cases = [
        {
            'url': 'https://example.com/fake-news-story',
            'description': 'URL with "fake" in path'
        },
        {
            'url': 'https://hoax-news.com/breaking-story',
            'description': 'URL with "hoax" in domain'
        },
        {
            'url': 'https://www.reuters.com/world/',
            'description': 'Legitimate Reuters URL'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"URL: {test_case['url']}")
        print("-" * 30)
        
        try:
            # Test just the keyword detection part
            content_data = detector.extract_content_for_analysis(test_case['url'])
            if content_data:
                keyword_result = detector.keyword_detection(test_case['url'], content_data)
                if keyword_result:
                    print(f"Keyword Detection: {keyword_result['prediction']} ({keyword_result['confidence']:.1%})")
                    if keyword_result['detected_keywords']:
                        print(f"Keywords found: {keyword_result['detected_keywords']}")
                    print(f"Fake score: {keyword_result['fake_score']}")
                else:
                    print("No keyword detection result")
            else:
                print("Could not extract content")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    test_fake_detection()