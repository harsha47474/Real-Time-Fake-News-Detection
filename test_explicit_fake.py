#!/usr/bin/env python3

"""
Test explicit fake content detection
"""

from ultimate_detector import UltimateDetector

def test_explicit_fake():
    print("🧪 Testing Explicit Fake Content Detection")
    print("=" * 50)
    
    detector = UltimateDetector()
    
    # Create a mock content with explicit "fake"
    mock_content = {
        'title': 'This is a fake news story',
        'content': 'This article contains fake information about recent events.',
        'full_text': 'this is a fake news story this article contains fake information about recent events.',
        'url_text': 'https://example.com/fake-news-story'
    }
    
    # Test keyword detection
    keyword_result = detector.keyword_detection('https://example.com/fake-news-story', mock_content)
    
    print("🔍 Keyword Detection Result:")
    print(f"   Prediction: {keyword_result['prediction']}")
    print(f"   Confidence: {keyword_result['confidence']:.1%}")
    print(f"   Fake Score: {keyword_result['fake_score']}")
    print(f"   Keywords: {keyword_result['detected_keywords']}")
    
    # Check if explicit fake is detected
    has_explicit = any('EXPLICIT' in kw for kw in keyword_result.get('detected_keywords', []))
    print(f"   Has Explicit 'fake': {has_explicit}")

if __name__ == '__main__':
    test_explicit_fake()