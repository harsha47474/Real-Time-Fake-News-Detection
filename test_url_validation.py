#!/usr/bin/env python3

"""
Test script to demonstrate URL validation
"""

import subprocess
import sys

def test_url(url, description):
    """Test a URL and show results"""
    print(f"\n🧪 Testing: {description}")
    print(f"📝 URL: {url}")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, 'ultimate_detector.py', '--url', url
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    print("🧪 URL Validation Test Suite")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("your_url_here", "Invalid placeholder URL"),
        ("not_a_url", "Invalid URL without protocol"),
        ("https://www.reuters.com/world/", "Valid Reuters URL"),
        ("https://www.bbc.com/news", "Valid BBC URL"),
    ]
    
    for url, description in test_cases:
        test_url(url, description)
    
    print("\n" + "=" * 60)
    print("✅ URL validation tests completed!")

if __name__ == '__main__':
    main()