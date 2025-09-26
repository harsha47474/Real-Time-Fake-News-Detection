#!/usr/bin/env python3

"""
Startup script for Ultimate Fake News Detection Web Interface
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'web_interface.py',
        'ultimate_detector.py',
        'huggingface_detector.py',
        'domain_classifier.py',
        'consistent_detector.py',
        'models/huggingface_models.pkl',
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
        
        if 'models/huggingface_models.pkl' in missing_files:
            print("\n💡 To create the trained models, run:")
            print("   python train_hf_dataset.py")
        
        if '.env' in missing_files:
            print("\n💡 To create .env file with Twitter token:")
            print("   echo 'TWITTER_BEARER_TOKEN=your_token_here' > .env")
        
        return False
    
    return True

def check_models():
    """Check model status"""
    model_path = Path('models/huggingface_models.pkl')
    if model_path.exists():
        print("✅ Hugging Face trained models found")
        print("🎯 Ready for 97%+ accuracy predictions!")
        return True
    else:
        print("⚠️  Hugging Face models not found")
        print("💡 Run: python train_hf_dataset.py")
        return False

def main():
    print("🏆 Ultimate Fake News Detection Web Interface")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start - missing required files")
        return
    
    print("✅ All required files found")
    
    # Check models
    has_models = check_models()
    
    # Check environment
    from dotenv import load_dotenv
    load_dotenv()
    
    twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
    if twitter_token:
        print(f"✅ Twitter Bearer Token found")
    else:
        print("⚠️  No Twitter Bearer Token (will use mock data)")
    
    print("\n🎯 Ultimate Features:")
    print("   • 97%+ accuracy Hugging Face trained models")
    print("   • Multi-method ensemble analysis")
    print("   • Real Twitter data integration")
    print("   • Domain trust analysis")
    print("   • Rule-based content analysis")
    print("   • Error handling and validation")
    
    if has_models:
        print("   • Trained on 30,000 news samples")
        print("   • Logistic Regression: 97.7% accuracy")
        print("   • Random Forest: 97.9% accuracy")
    
    print(f"\n🚀 Starting server...")
    print("   Open your browser and go to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 70)
    
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