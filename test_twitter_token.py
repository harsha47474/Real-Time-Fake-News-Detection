#!/usr/bin/env python3

"""
Test script to verify Twitter Bearer Token
"""

import os
import tweepy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_twitter_token():
    """Test if the Twitter Bearer Token is working"""
    print("🐦 Testing Twitter Bearer Token...")
    
    # Get token from environment
    token = os.getenv('TWITTER_BEARER_TOKEN')
    
    if not token:
        print("❌ No Twitter Bearer Token found in environment variables")
        print("   Make sure TWITTER_BEARER_TOKEN is set in your .env file")
        return False
    
    print(f"✅ Token found: {token[:20]}...")
    
    try:
        # Initialize Twitter client
        client = tweepy.Client(bearer_token=token)
        
        # Test with a simple search
        print("🔍 Testing API with a simple search...")
        
        tweets = client.search_recent_tweets(
            query="news -is:retweet lang:en",
            max_results=10,  # Minimum allowed by Twitter API
            tweet_fields=['created_at', 'author_id', 'public_metrics', 'text']
        )
        
        if tweets.data:
            print(f"✅ API working! Found {len(tweets.data)} tweets")
            
            for i, tweet in enumerate(tweets.data[:3], 1):
                print(f"\n📰 Tweet {i}:")
                print(f"   Text: {tweet.text[:100]}...")
                print(f"   Likes: {tweet.public_metrics['like_count']}")
                print(f"   Retweets: {tweet.public_metrics['retweet_count']}")
                
            return True
        else:
            print("⚠️ API responded but no tweets found")
            return False
            
    except tweepy.TooManyRequests:
        print("❌ Rate limit exceeded - API is working but you've hit the limit")
        print("   Wait 15 minutes and try again")
        return False
        
    except tweepy.Unauthorized:
        print("❌ Unauthorized - Invalid Bearer Token")
        print("   Check if your token is correct")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_specific_search(domain="reuters.com"):
    """Test searching for a specific domain"""
    token = os.getenv('TWITTER_BEARER_TOKEN')
    if not token:
        return
    
    try:
        client = tweepy.Client(bearer_token=token)
        
        print(f"\n🔍 Testing search for domain: {domain}")
        
        # Try different search queries
        queries = [
            f"{domain} -is:retweet lang:en",
            f"url:{domain} -is:retweet",
            f"reuters -is:retweet lang:en"  # More general search
        ]
        
        for query in queries:
            print(f"\n   Query: {query}")
            try:
                tweets = client.search_recent_tweets(
                    query=query,
                    max_results=10,  # Minimum allowed
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if tweets.data:
                    print(f"   ✅ Found {len(tweets.data)} tweets")
                else:
                    print(f"   ⚠️ No tweets found")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
    
    except Exception as e:
        print(f"❌ Error in specific search: {e}")

if __name__ == '__main__':
    print("🚀 Twitter API Test")
    print("=" * 40)
    
    # Test basic functionality
    if test_twitter_token():
        # Test specific domain search
        test_specific_search()
    
    print("\n" + "=" * 40)
    print("✅ Test completed!")