#!/usr/bin/env python3

"""
Test the ultimate detector for web interface
"""

from ultimate_detector import UltimateDetector
import json

def test_detector():
    print("🧪 Testing Ultimate Detector for Web Interface")
    print("=" * 50)
    
    # Initialize detector
    detector = UltimateDetector()
    
    # Test URLs
    test_urls = [
        "https://www.reuters.com/world/",
        "https://www.bbc.com/news",
        "invalid_url",
        "https://timesofindia.indiatimes.com/business/india-business/trumps-100-tariffs-on-pharma-sun-pharma-biocon-cipla-other-pharmaceutical-stocks-tank-jitters-on-d-street/articleshow/124144816.cms"
    ]
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n🔍 Test {i}: {url}")
        print("-" * 30)
        
        try:
            result = detector.predict(url)
            
            # Check result structure
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success: {result['final_prediction']['label']} ({result['final_prediction']['confidence']:.1%})")
                print(f"📰 Title: {result['title']}")
                print(f"🌐 Domain: {result['domain']}")
                
                # Check structure for web interface
                required_keys = ['url', 'title', 'domain', 'final_prediction', 'method']
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    print(f"⚠️  Missing keys: {missing_keys}")
                else:
                    print("✅ All required keys present")
            
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n✅ Testing completed!")

if __name__ == '__main__':
    test_detector()