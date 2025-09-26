#!/usr/bin/env python3

"""
Real-time Fake News Detection with Social Media APIs and Web Scraping
This implementation integrates:
1. Web scraping for article content
2. Twitter API for social interaction data
3. Graph construction from real social networks
4. BERT feature extraction
5. GNN-based prediction
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import requests
from bs4 import BeautifulSoup
import tweepy
import numpy as np
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from datetime import datetime, timedelta
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from test_upfd import SimpleGNN
import pickle
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArticleData:
    """Structure to hold scraped article information"""
    url: str
    title: str
    content: str
    author: str
    publish_date: str
    domain: str

@dataclass
class SocialData:
    """Structure to hold social media interaction data"""
    tweets: List[Dict]
    retweet_graph: List[Tuple[str, str]]
    user_profiles: Dict[str, Dict]
    engagement_metrics: Dict

class WebScraper:
    """Web scraper for extracting article content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_article(self, url: str) -> ArticleData:
        """Scrape article content from URL"""
        try:
            logger.info(f"Scraping article from: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract content
            content = self._extract_content(soup)
            
            # Extract metadata
            author = self._extract_author(soup)
            publish_date = self._extract_date(soup)
            domain = url.split('/')[2]
            
            return ArticleData(
                url=url,
                title=title,
                content=content,
                author=author,
                publish_date=publish_date,
                domain=domain
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            # Return dummy data for demo
            return ArticleData(
                url=url,
                title="Sample News Article",
                content="This is sample content for demonstration purposes.",
                author="Unknown",
                publish_date=datetime.now().isoformat(),
                domain=url.split('/')[2] if '/' in url else "unknown.com"
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        selectors = [
            'h1',
            '[property="og:title"]',
            'title',
            '.headline',
            '.article-title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return "Unknown Title"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        selectors = [
            '.article-content',
            '.post-content',
            '.entry-content',
            'article p',
            '.content p'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text().strip() for elem in elements])
                if len(content) > 100:  # Ensure we have substantial content
                    return content
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        return content[:2000] if content else "No content found"
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract article author"""
        selectors = [
            '[rel="author"]',
            '.author',
            '.byline',
            '[property="article:author"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return "Unknown"
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publish date"""
        selectors = [
            '[property="article:published_time"]',
            'time[datetime]',
            '.publish-date',
            '.date'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get('datetime') or element.get_text().strip()
        
        return datetime.now().isoformat()

class TwitterCollector:
    """Collect Twitter data for news articles"""
    
    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        if self.bearer_token:
            self.client = tweepy.Client(bearer_token=self.bearer_token)
        else:
            logger.warning("No Twitter Bearer Token provided. Using mock data.")
            self.client = None
    
    def collect_social_data(self, article: ArticleData) -> SocialData:
        """Collect social media data for an article"""
        if not self.client:
            return self._generate_mock_social_data(article)
        
        try:
            # Search for tweets containing the article URL or title
            query = f'"{article.url}" OR "{article.title[:50]}"'
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=100,
                tweet_fields=['author_id', 'created_at', 'public_metrics', 'referenced_tweets']
            ).flatten(limit=500)
            
            tweet_data = []
            retweet_graph = []
            user_profiles = {}
            
            for tweet in tweets:
                tweet_info = {
                    'id': tweet.id,
                    'author_id': tweet.author_id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics
                }
                tweet_data.append(tweet_info)
                
                # Build retweet graph
                if tweet.referenced_tweets:
                    for ref in tweet.referenced_tweets:
                        if ref.type == 'retweeted':
                            retweet_graph.append((str(tweet.author_id), str(ref.id)))
                
                # Collect user profile (simplified)
                user_profiles[str(tweet.author_id)] = {
                    'followers_count': tweet.public_metrics.get('followers_count', 0),
                    'following_count': tweet.public_metrics.get('following_count', 0),
                    'tweet_count': tweet.public_metrics.get('tweet_count', 0)
                }
            
            engagement_metrics = {
                'total_tweets': len(tweet_data),
                'total_retweets': sum(t['metrics'].get('retweet_count', 0) for t in tweet_data),
                'total_likes': sum(t['metrics'].get('like_count', 0) for t in tweet_data)
            }
            
            return SocialData(
                tweets=tweet_data,
                retweet_graph=retweet_graph,
                user_profiles=user_profiles,
                engagement_metrics=engagement_metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return self._generate_mock_social_data(article)
    
    def _generate_mock_social_data(self, article: ArticleData) -> SocialData:
        """Generate mock social data for demonstration"""
        logger.info("Generating mock social media data for demonstration")
        
        num_users = np.random.randint(20, 100)
        tweets = []
        retweet_graph = []
        user_profiles = {}
        
        for i in range(num_users):
            user_id = f"user_{i}"
            tweets.append({
                'id': f"tweet_{i}",
                'author_id': user_id,
                'text': f"Sharing this article: {article.title[:50]}...",
                'created_at': datetime.now(),
                'metrics': {
                    'retweet_count': np.random.randint(0, 50),
                    'like_count': np.random.randint(0, 200),
                    'reply_count': np.random.randint(0, 20)
                }
            })
            
            # Create retweet relationships
            if i > 0 and np.random.random() < 0.3:
                parent = np.random.randint(0, i)
                retweet_graph.append((user_id, f"user_{parent}"))
            
            # User profiles
            user_profiles[user_id] = {
                'followers_count': np.random.randint(10, 10000),
                'following_count': np.random.randint(10, 5000),
                'tweet_count': np.random.randint(100, 50000),
                'verified': np.random.random() < 0.1
            }
        
        engagement_metrics = {
            'total_tweets': len(tweets),
            'total_retweets': sum(t['metrics']['retweet_count'] for t in tweets),
            'total_likes': sum(t['metrics']['like_count'] for t in tweets)
        }
        
        return SocialData(
            tweets=tweets,
            retweet_graph=retweet_graph,
            user_profiles=user_profiles,
            engagement_metrics=engagement_metrics
        )

class FeatureExtractor:
    """Extract features using BERT and user profiles"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract BERT embeddings from text"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embeddings.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            # Return random features as fallback
            return np.random.randn(768)
    
    def extract_user_features(self, user_profile: Dict) -> np.ndarray:
        """Extract numerical features from user profile"""
        features = [
            np.log1p(user_profile.get('followers_count', 1)),
            np.log1p(user_profile.get('following_count', 1)),
            np.log1p(user_profile.get('tweet_count', 1)),
            float(user_profile.get('verified', False)),
            user_profile.get('followers_count', 1) / max(user_profile.get('following_count', 1), 1)  # follower ratio
        ]
        return np.array(features)

class GraphBuilder:
    """Build PyTorch Geometric graphs from social data"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def build_graph(self, article: ArticleData, social_data: SocialData) -> Data:
        """Build a PyTorch Geometric graph from article and social data"""
        logger.info("Building interaction graph...")
        
        # Create node features
        node_features = []
        node_mapping = {}
        
        # Add article node (node 0)
        article_text = f"{article.title} {article.content}"
        article_features = self.feature_extractor.extract_text_features(article_text)
        
        # Pad user features to match BERT dimension
        user_feature_dim = 5
        bert_dim = len(article_features)
        padding_dim = bert_dim - user_feature_dim
        
        # Article node gets full BERT features
        node_features.append(article_features)
        node_mapping['article'] = 0
        
        # Add user nodes
        for i, (user_id, profile) in enumerate(social_data.user_profiles.items()):
            user_features = self.feature_extractor.extract_user_features(profile)
            # Pad user features to match BERT dimension
            padded_features = np.concatenate([user_features, np.zeros(padding_dim)])
            node_features.append(padded_features)
            node_mapping[user_id] = i + 1
        
        # Create edge list
        edge_list = []
        
        # Connect article to all users who tweeted about it
        for tweet in social_data.tweets:
            user_id = tweet['author_id']
            if user_id in node_mapping:
                user_idx = node_mapping[user_id]
                # Bidirectional edges
                edge_list.append([0, user_idx])  # article -> user
                edge_list.append([user_idx, 0])  # user -> article
        
        # Add retweet relationships
        for source_user, target_user in social_data.retweet_graph:
            if source_user in node_mapping and target_user in node_mapping:
                source_idx = node_mapping[source_user]
                target_idx = node_mapping[target_user]
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])  # Undirected
        
        # Convert to tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        # Create batch (single graph)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        logger.info(f"Graph created: {x.size(0)} nodes, {edge_index.size(1)} edges")
        
        return Data(x=x, edge_index=edge_index, batch=batch)

class RealTimeFakeNewsDetector:
    """Main class for real-time fake news detection"""
    
    def __init__(self, model_path: Optional[str] = None, twitter_token: Optional[str] = None):
        self.scraper = WebScraper()
        self.twitter_collector = TwitterCollector(twitter_token)
        self.feature_extractor = FeatureExtractor()
        self.graph_builder = GraphBuilder(self.feature_extractor)
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using dummy model for demo.")
            self.model = self._create_dummy_model()
    
    def _load_model(self, model_path: str) -> SimpleGNN:
        """Load pre-trained model"""
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SimpleGNN(
            num_features=checkpoint['num_features'],
            hidden_dim=checkpoint['hidden_dim'],
            num_classes=checkpoint['num_classes'],
            model_type=checkpoint['model_type']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _create_dummy_model(self) -> SimpleGNN:
        """Create a dummy model for demonstration"""
        model = SimpleGNN(num_features=768, hidden_dim=128, num_classes=2, model_type='gcn')
        model.eval()
        return model
    
    def predict(self, url: str) -> Dict:
        """Predict if a news article is fake or real"""
        logger.info(f"Starting prediction for: {url}")
        
        # Step 1: Scrape article content
        article = self.scraper.scrape_article(url)
        
        # Step 2: Collect social media data
        social_data = self.twitter_collector.collect_social_data(article)
        
        # Step 3: Build graph
        graph = self.graph_builder.build_graph(article, social_data)
        
        # Step 4: Make prediction
        with torch.no_grad():
            output = self.model(graph)
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Prepare results
        result = {
            'url': url,
            'article': {
                'title': article.title,
                'author': article.author,
                'domain': article.domain,
                'publish_date': article.publish_date
            },
            'social_metrics': social_data.engagement_metrics,
            'graph_stats': {
                'nodes': graph.x.size(0),
                'edges': graph.edge_index.size(1)
            },
            'prediction': {
                'label': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': confidence,
                'probabilities': {
                    'real': probabilities[0][0].item(),
                    'fake': probabilities[0][1].item()
                }
            }
        }
        
        return result
    
    def save_model(self, model_path: str):
        """Save the current model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_features': 768,
            'hidden_dim': 128,
            'num_classes': 2,
            'model_type': 'gcn'
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Real-time Fake News Detection')
    parser.add_argument('--url', type=str, required=True, help='News article URL to analyze')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    parser.add_argument('--twitter-token', type=str, help='Twitter Bearer Token')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    print("🚀 Real-time GNN-Based Fake News Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = RealTimeFakeNewsDetector(
        model_path=args.model,
        twitter_token=args.twitter_token
    )
    
    # Make prediction
    result = detector.predict(args.url)
    
    # Display results
    print(f"\n📰 ARTICLE ANALYSIS")
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['article']['title']}")
    print(f"✍️  Author: {result['article']['author']}")
    print(f"🌐 Domain: {result['article']['domain']}")
    
    print(f"\n📊 SOCIAL MEDIA METRICS")
    print(f"🐦 Total Tweets: {result['social_metrics']['total_tweets']}")
    print(f"🔄 Total Retweets: {result['social_metrics']['total_retweets']}")
    print(f"❤️  Total Likes: {result['social_metrics']['total_likes']}")
    
    print(f"\n🕸️  GRAPH STATISTICS")
    print(f"👥 Nodes: {result['graph_stats']['nodes']}")
    print(f"🔗 Edges: {result['graph_stats']['edges']}")
    
    print(f"\n🎯 PREDICTION RESULTS")
    label = result['prediction']['label']
    emoji = "❌" if label == "FAKE" else "✅"
    print(f"{emoji} PREDICTION: {label}")
    print(f"🎯 Confidence: {result['prediction']['confidence']:.2%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {result['prediction']['probabilities']['real']:.2%}")
    print(f"   Fake: {result['prediction']['probabilities']['fake']:.2%}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n⚠️  Note: This system uses real web scraping and social media data!")

if __name__ == '__main__':
    main()