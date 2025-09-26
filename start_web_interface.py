#!/usr/bin/env python3

"""
Startup script for the Fixed Fake News Detection Web Interface
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'web_interface.py',
        'fixed_improved_detector.py',
        'consistent_detector.py',
        'domain_classifier.py',
        '.env'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_environment():
    """Check environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
    if twitter_token:
        print(f"✅ Twitter Bearer Token found: {twitter_token[:20]}...")
    else:
        print("⚠️  No Twitter Bearer Token found (will use mock data)")
    
    return True

def main():
    print("🚀 Starting Fixed Fake News Detection Web Interface")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start - missing required files")
        return
    
    print("✅ All required files found")
    
    # Check environment
    check_environment()
    
    print("\n🎯 Features:")
    print("   • Consistent results every time")
    print("   • Multi-method ensemble analysis")
    print("   • Real Twitter data integration")
    print("   • Consistency testing")
    print("   • Batch processing")
    
    print(f"\n🔗 Starting server...")
    print("   Open your browser and go to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run the web interface
    try:
        from web_interface import main as run_web_interface
        run_web_interface()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Try running: python web_interface.py")

if __name__ == '__main__':
    main()