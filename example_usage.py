#!/usr/bin/env python3

"""
Example Usage Script for Real-time Fake News Detection
This script demonstrates various ways to use the system
"""

import os
import time
from real_time_detector import RealTimeFakeNewsDetector
from batch_detector import BatchProcessor
import json

def example_single_url():
    """Example: Analyze a single news URL"""
    print("🔍 Example 1: Single URL Analysis")
    print("=" * 40)
    
    # Initialize detector
    detector = RealTimeFakeNewsDetector()
    
    # Example URLs (you can replace with real news URLs)
    test_urls = [
        "https://www.reuters.com/world/us/",
        "https://www.bbc.com/news",
        "https://edition.cnn.com/",
        "https://www.nytimes.com/"
    ]
    
    for url in test_urls[:2]:  # Test first 2 URLs
        print(f"\n📰 Analyzing: {url}")
        try:
            result = detector.predict(url)
            
            print(f"🎯 Prediction: {result['prediction']['label']}")
            print(f"📊 Confidence: {result['prediction']['confidence']:.1%}")
            print(f"📰 Title: {result['article']['title']}")
            print(f"🌐 Domain: {result['article']['domain']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Be respectful to websites

def example_batch_processing():
    """Example: Batch process multiple URLs"""
    print("\n📊 Example 2: Batch Processing")
    print("=" * 40)
    
    # Create sample URL file
    sample_urls = [
        "https://www.reuters.com/world/",
        "https://www.bbc.com/news/world",
        "https://edition.cnn.com/politics",
        "https://www.npr.org/sections/news/"
    ]
    
    # Write URLs to file
    with open('sample_urls.txt', 'w') as f:
        for url in sample_urls:
            f.write(url + '\n')
    
    print(f"📝 Created sample file with {len(sample_urls)} URLs")
    
    # Initialize batch processor
    processor = BatchProcessor(max_workers=2)
    
    try:
        # Load and process URLs
        urls = processor.load_urls('sample_urls.txt')
        print(f"📋 Loaded {len(urls)} URLs")
        
        # Process URLs (sequential for demo)
        results = processor.process_urls(urls, parallel=False)
        
        # Generate statistics
        stats = processor.generate_summary_stats()
        
        print(f"\n📈 Results Summary:")
        print(f"✅ Successful: {stats['successful_predictions']}")
        print(f"❌ Errors: {stats['errors']}")
        print(f"🎯 Fake predictions: {stats['predictions']['fake']}")
        print(f"✅ Real predictions: {stats['predictions']['real']}")
        
        # Save results
        output_files = processor.save_results('example_results')
        print(f"\n💾 Results saved to: example_results/")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    # Cleanup
    if os.path.exists('sample_urls.txt'):
        os.remove('sample_urls.txt')

def example_with_twitter_api():
    """Example: Using Twitter API (requires token)"""
    print("\n🐦 Example 3: With Twitter API")
    print("=" * 40)
    
    twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
    
    if not twitter_token:
        print("⚠️  No Twitter Bearer Token found in environment variables")
        print("   Set TWITTER_BEARER_TOKEN to use real Twitter data")
        print("   For now, the system will use mock social media data")
    else:
        print("✅ Twitter API token found - will use real social data")
    
    # Initialize detector with Twitter token
    detector = RealTimeFakeNewsDetector(twitter_token=twitter_token)
    
    # Test URL
    test_url = "https://www.reuters.com/world/"
    
    print(f"\n📰 Analyzing with social data: {test_url}")
    
    try:
        result = detector.predict(test_url)
        
        print(f"\n📊 Social Media Metrics:")
        print(f"🐦 Tweets: {result['social_metrics']['total_tweets']}")
        print(f"🔄 Retweets: {result['social_metrics']['total_retweets']}")
        print(f"❤️  Likes: {result['social_metrics']['total_likes']}")
        
        print(f"\n🕸️  Graph Statistics:")
        print(f"👥 Nodes: {result['graph_stats']['nodes']}")
        print(f"🔗 Edges: {result['graph_stats']['edges']}")
        
        print(f"\n🎯 Prediction:")
        print(f"{result['prediction']['label']} ({result['prediction']['confidence']:.1%} confidence)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_custom_configuration():
    """Example: Custom configuration and model"""
    print("\n⚙️  Example 4: Custom Configuration")
    print("=" * 40)
    
    # Check if custom model exists
    model_path = "models/custom_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Loading custom model: {model_path}")
        detector = RealTimeFakeNewsDetector(model_path=model_path)
    else:
        print("⚠️  No custom model found, using default")
        detector = RealTimeFakeNewsDetector()
    
    # Example with configuration
    test_url = "https://www.bbc.com/news"
    
    print(f"\n📰 Analyzing: {test_url}")
    
    try:
        result = detector.predict(test_url)
        
        # Save detailed results
        output_file = "detailed_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"🎯 Quick result: {result['prediction']['label']} ({result['prediction']['confidence']:.1%})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_web_interface_info():
    """Example: Information about web interface"""
    print("\n🌐 Example 5: Web Interface")
    print("=" * 40)
    
    print("To use the web interface:")
    print("1. Run: python web_interface.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Enter news URLs for analysis")
    print("4. View results in real-time")
    
    print("\nWeb interface features:")
    print("✅ Single URL analysis")
    print("✅ Batch processing (up to 10 URLs)")
    print("✅ Visual results with confidence scores")
    print("✅ Article information display")
    print("✅ Social media metrics")

def main():
    """Run all examples"""
    print("🚀 GNN Fake News Detection - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_single_url()
        example_batch_processing()
        example_with_twitter_api()
        example_custom_configuration()
        example_web_interface_info()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   - Set up Twitter API for real social data")
        print("   - Train custom models on your data")
        print("   - Use the web interface for easy analysis")
        print("   - Process large batches of URLs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")

if __name__ == '__main__':
    main()