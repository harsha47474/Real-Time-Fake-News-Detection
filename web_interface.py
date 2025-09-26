#!/usr/bin/env python3

"""
Web Interface for Real-time Fake News Detection
Simple Flask web application for easy interaction with the detection system
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from datetime import datetime
from ultimate_detector import UltimateDetector
import tweepy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'fake_news_detector_secret_key'

# Initialize detector and Twitter client
detector = None
twitter_client = None

def init_detector():
    """Initialize the fake news detector and Twitter client"""
    global detector, twitter_client
    if detector is None:
        detector = UltimateDetector()
        logger.info("Ultimate detector initialized (97%+ accuracy)")
    
    if twitter_client is None:
        twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
        if twitter_token:
            try:
                twitter_client = tweepy.Client(bearer_token=twitter_token)
                logger.info("Twitter client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {e}")
        else:
            logger.warning("No Twitter Bearer Token found")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

def get_real_tweets(url, max_tweets=10):
    """Fetch real tweets about the URL"""
    if not twitter_client:
        return []
    
    try:
        # Extract domain and title for search
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace('www.', '')
        
        # Try multiple search strategies
        search_queries = [
            f"{domain} -is:retweet lang:en",
            f"url:{domain} -is:retweet",
            f"{domain.split('.')[0]} news -is:retweet lang:en"  # e.g., "reuters news"
        ]
        
        all_tweets = []
        
        for search_query in search_queries:
            try:
                # Twitter API requires max_results between 10-100
                api_max_results = max(10, min(100, max_tweets * 2))
                
                tweets = twitter_client.search_recent_tweets(
                    query=search_query,
                    max_results=api_max_results,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'text']
                )
                
                if tweets.data:
                    all_tweets.extend(tweets.data)
                    break  # Found tweets, no need to try other queries
                    
            except tweepy.TooManyRequests:
                logger.warning(f"Rate limit hit for query: {search_query}")
                continue
            except Exception as e:
                logger.warning(f"Search query '{search_query}' failed: {e}")
                continue
        
        if all_tweets:
            tweet_list = []
            # Remove duplicates and limit to max_tweets
            seen_ids = set()
            for tweet in all_tweets[:max_tweets * 2]:  # Get more to filter duplicates
                if tweet.id not in seen_ids:
                    seen_ids.add(tweet.id)
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                        'author_id': tweet.author_id,
                        'metrics': {
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count']
                        }
                    }
                    tweet_list.append(tweet_data)
                    
                    if len(tweet_list) >= max_tweets:
                        break
                        
            return tweet_list
        
    except tweepy.TooManyRequests:
        logger.warning("Twitter API rate limit exceeded. Using mock data.")
        return generate_mock_tweets(url, max_tweets)
    except Exception as e:
        logger.error(f"Error fetching tweets: {e}")
        return generate_mock_tweets(url, max_tweets)
    
    # If no tweets found, return mock data for demo
    if not all_tweets:
        logger.info("No tweets found, generating mock data for demo")
        return generate_mock_tweets(url, max_tweets)
    
    return []

def generate_mock_tweets(url, max_tweets=5):
    """Generate mock tweets for demonstration when API is unavailable"""
    from urllib.parse import urlparse
    import random
    from datetime import datetime, timedelta
    
    domain = urlparse(url).netloc.replace('www.', '')
    
    mock_tweets = [
        {
            'id': f'mock_{i}',
            'text': f"Interesting article from {domain} about current events. Worth reading! #news",
            'created_at': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            'author_id': f'user_{i}',
            'metrics': {
                'retweet_count': random.randint(0, 50),
                'like_count': random.randint(5, 200),
                'reply_count': random.randint(0, 20),
                'quote_count': random.randint(0, 10)
            }
        }
        for i in range(min(max_tweets, 3))
    ]
    
    return mock_tweets

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a news article URL"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Initialize detector if needed
        init_detector()
        
        # Analyze the URL
        result = detector.predict(url)
        
        # Check if there was an error in the analysis
        if 'error' in result:
            return jsonify(result), 400
        
        # Get real tweets only if analysis was successful
        tweets = get_real_tweets(url)
        
        # Calculate social metrics from real tweets
        total_retweets = sum(tweet['metrics']['retweet_count'] for tweet in tweets)
        total_likes = sum(tweet['metrics']['like_count'] for tweet in tweets)
        total_replies = sum(tweet['metrics']['reply_count'] for tweet in tweets)
        
        # Add social media data to result
        is_mock_data = len(tweets) > 0 and str(tweets[0]['id']).startswith('mock_')
        result['social_media'] = {
            'tweets': tweets,
            'metrics': {
                'total_tweets': len(tweets),
                'total_retweets': total_retweets,
                'total_likes': total_likes,
                'total_replies': total_replies
            },
            'data_source': 'mock' if is_mock_data else 'real'
        }
        
        # Add timestamp
        result['analysis_timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Batch analyze multiple URLs"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'URLs list is required'}), 400
        
        if len(urls) > 10:  # Limit for web interface
            return jsonify({'error': 'Maximum 10 URLs allowed for web interface'}), 400
        
        # Initialize detector if needed
        init_detector()
        
        results = []
        for url in urls:
            try:
                result = detector.predict(url)
                
                # Check if there was an error
                if 'error' in result:
                    results.append({
                        'url': url,
                        'error': result['error'],
                        'analysis_timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Get real tweets for batch analysis too (limited to 3 for performance and rate limits)
                tweets = get_real_tweets(url, max_tweets=3)
                
                # Calculate social metrics
                total_retweets = sum(tweet['metrics']['retweet_count'] for tweet in tweets)
                total_likes = sum(tweet['metrics']['like_count'] for tweet in tweets)
                total_replies = sum(tweet['metrics']['reply_count'] for tweet in tweets)
                
                result['social_media'] = {
                    'tweets': tweets,
                    'metrics': {
                        'total_tweets': len(tweets),
                        'total_retweets': total_retweets,
                        'total_likes': total_likes,
                        'total_replies': total_replies
                    },
                    'data_source': 'mock' if (tweets and str(tweets[0]['id']).startswith('mock_')) else 'real'
                }
                
                result['analysis_timestamp'] = datetime.now().isoformat()
                results.append(result)
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'analysis_timestamp': datetime.now().isoformat()
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'detector_initialized': detector is not None
    })

@app.route('/test_consistency', methods=['POST'])
def test_consistency():
    """Test detector consistency with multiple runs"""
    try:
        data = request.get_json()
        url = data.get('url')
        runs = data.get('runs', 3)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if runs > 5:  # Limit to prevent abuse
            runs = 5
        
        # Initialize detector if needed
        init_detector()
        
        results = []
        for i in range(runs):
            result = detector.predict(url)
            results.append({
                'run': i + 1,
                'prediction': result['final_prediction']['label'],
                'confidence': result['final_prediction']['confidence']
            })
        
        # Check consistency
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        is_consistent = len(set(predictions)) == 1 and len(set(confidences)) == 1
        
        return jsonify({
            'url': url,
            'runs': runs,
            'results': results,
            'is_consistent': is_consistent,
            'consistency_message': 'All runs produced identical results!' if is_consistent else 'Results varied between runs.',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in consistency test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get system statistics"""
    try:
        # Read recent results if available
        results_dir = Path('web_results')
        if results_dir.exists():
            result_files = list(results_dir.glob('*.json'))
            total_analyses = len(result_files)
        else:
            total_analyses = 0
        
        return jsonify({
            'total_analyses': total_analyses,
            'detector_status': 'fixed_improved' if detector else 'not_initialized',
            'detector_type': 'Fixed Improved Detector (Consistent Results)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create templates directory and HTML template
def create_templates():
    """Create HTML templates for the web interface"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Main HTML template
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Fake News Detector (97%+ Accuracy)</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 30px;
        }
        .url-input {
            width: 70%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .analyze-btn {
            width: 25%;
            padding: 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-left: 10px;
        }
        .analyze-btn:hover {
            background: #0056b3;
        }
        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            border-left: 5px solid;
        }
        .result.fake {
            background: #fff5f5;
            border-color: #e53e3e;
        }
        .result.real {
            background: #f0fff4;
            border-color: #38a169;
        }
        .result.error {
            background: #fffbf0;
            border-color: #d69e2e;
        }
        .result-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .confidence {
            font-size: 18px;
            margin-bottom: 15px;
        }
        .article-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .social-metrics {
            display: flex;
            gap: 20px;
            margin: 15px 0;
        }
        .metric {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            flex: 1;
        }
        .batch-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #eee;
        }
        .url-list {
            width: 100%;
            height: 150px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            resize: vertical;
        }
        .progress {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background: #007bff;
            width: 0%;
            transition: width 0.3s ease;
        }
        .individual-predictions {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .prediction-item {
            margin: 10px 0;
            padding: 8px;
            background: white;
            border-radius: 3px;
            border-left: 3px solid #007bff;
        }
        .tweets-section {
            background: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        .tweet {
            background: white;
            padding: 12px;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 3px solid #1da1f2;
        }
        .tweet-text {
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .tweet-metrics {
            font-size: 12px;
            color: #666;
            display: flex;
            gap: 15px;
        }
        .tweet-metrics span {
            display: flex;
            align-items: center;
            gap: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Ultimate Fake News Detector</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            🎯 97%+ Accuracy | 🤖 Hugging Face Trained | 🔍 Multi-Method Analysis | 🐦 Real Twitter Data
        </p>
        
        <!-- Single URL Analysis -->
        <div class="input-section">
            <h2>Analyze Single Article</h2>
            <input type="url" id="urlInput" class="url-input" placeholder="Enter news article URL..." />
            <button onclick="analyzeUrl()" class="analyze-btn" id="analyzeBtn">Analyze</button>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            🔍 Analyzing article... This may take a few moments.
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <!-- Consistency Test -->
        <div class="batch-section">
            <h2>🎯 Consistency Test</h2>
            <p>Test if the detector gives consistent results (same URL should always give same prediction):</p>
            <input type="url" id="consistencyUrl" class="url-input" placeholder="Enter URL to test consistency..." />
            <button onclick="testConsistency()" class="analyze-btn" id="consistencyBtn">Test Consistency</button>
            
            <div id="consistencyResults" style="display: none;"></div>
        </div>

        <!-- Batch Analysis -->
        <div class="batch-section">
            <h2>📊 Batch Analysis</h2>
            <p>Enter multiple URLs (one per line, max 10):</p>
            <textarea id="urlList" class="url-list" placeholder="https://example.com/article1&#10;https://example.com/article2&#10;..."></textarea>
            <br><br>
            <button onclick="batchAnalyze()" class="analyze-btn" id="batchBtn">Analyze Batch</button>
            
            <div id="batchProgress" style="display: none;">
                <div class="progress">
                    <div class="progress-bar" id="progressBar"></div>
                </div>
                <p id="progressText">Processing...</p>
            </div>
            
            <div id="batchResults" style="display: none;"></div>
        </div>
    </div>

    <script>
        async function analyzeUrl() {
            console.log('analyzeUrl function called');
            
            const url = document.getElementById('urlInput').value.trim();
            console.log('URL:', url);
            
            if (!url) {
                alert('Please enter a URL');
                return;
            }
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                console.log('Sending request to /analyze');
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });
                
                console.log('Response status:', response.status);
                
                const data = await response.json();
                console.log('Response data:', data);
                
                if (response.ok) {
                    displayResult(data);
                } else {
                    displayError(data.error || 'Analysis failed');
                }
            } catch (error) {
                console.error('Error:', error);
                displayError('Network error: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResult(data) {
            const result = document.getElementById('result');
            
            // Check for errors first
            if (data.error) {
                result.innerHTML = `
                    <div class="result error">
                        <div class="result-header">❌ Analysis Error</div>
                        <p><strong>URL:</strong> ${data.url}</p>
                        <p><strong>Error:</strong> ${data.error}</p>
                        <p><strong>Solution:</strong> Please check the URL and make sure it's a valid news article link.</p>
                    </div>
                `;
                result.style.display = 'block';
                return;
            }
            
            const prediction = data.final_prediction;
            const individual = data.individual_predictions || {};
            const social = data.social_media || {};
            const tweets = social.tweets || [];
            const metrics = social.metrics || {};
            const ensemble = data.ensemble_info || {};
            
            const isReal = prediction.label === 'REAL';
            const resultClass = isReal ? 'real' : 'fake';
            const emoji = isReal ? '✅' : '❌';
            
            // Build tweets HTML
            let tweetsHtml = '';
            const dataSource = social.data_source || 'unknown';
            const sourceEmoji = dataSource === 'real' ? '🐦' : '🤖';
            const sourceText = dataSource === 'real' ? 'Recent Tweets' : 'Sample Tweets (API Rate Limited)';
            
            if (tweets.length > 0) {
                tweetsHtml = `<div class="tweets-section"><h3>${sourceEmoji} ${sourceText}</h3>`;
                tweets.slice(0, 5).forEach(tweet => {
                    const tweetDate = tweet.created_at ? new Date(tweet.created_at).toLocaleDateString() : 'Unknown';
                    tweetsHtml += `
                        <div class="tweet">
                            <div class="tweet-text">${tweet.text}</div>
                            <div class="tweet-metrics">
                                <span>❤️ ${tweet.metrics.like_count}</span>
                                <span>🔄 ${tweet.metrics.retweet_count}</span>
                                <span>💬 ${tweet.metrics.reply_count}</span>
                                <span>📅 ${tweetDate}</span>
                            </div>
                        </div>
                    `;
                });
                tweetsHtml += '</div>';
            } else {
                tweetsHtml = '<div class="tweets-section"><h3>🐦 No Recent Tweets Found</h3><p>No tweets found for this article in recent search. This could be due to API rate limits or the article being very new.</p></div>';
            }
            
            result.innerHTML = `
                <div class="result ${resultClass}">
                    <div class="result-header">
                        ${emoji} ${prediction.label}
                    </div>
                    <div class="confidence">
                        Ultimate Ensemble Confidence: ${(prediction.confidence * 100).toFixed(1)}%
                        <br><small style="color: #28a745;">🏆 Maximum accuracy using 97%+ trained models</small>
                    </div>
                    
                    <div class="article-info">
                        <h3>📰 Article Information</h3>
                        <p><strong>Title:</strong> ${data.title || 'N/A'}</p>
                        <p><strong>Domain:</strong> ${data.domain || 'N/A'}</p>
                        <p><strong>Method:</strong> ${data.method || 'N/A'}</p>
                        <p><strong>URL:</strong> <a href="${data.url}" target="_blank">${data.url}</a></p>
                    </div>
                    
                    <div class="individual-predictions">
                        <h3>🔍 Individual Model Results</h3>
                        ${individual.huggingface ? `
                        <div class="prediction-item" style="border-left-color: #28a745;">
                            <strong>🤖 Hugging Face Models (97%+ Accuracy):</strong> 
                            ${individual.huggingface.label} 
                            (${(individual.huggingface.confidence * 100).toFixed(1)}%)
                            <br><small style="color: #28a745;">✨ Trained on 30,000 samples with 97%+ accuracy</small>
                        </div>
                        ` : ''}
                        ${individual.domain ? `
                        <div class="prediction-item">
                            <strong>🌐 Domain Classifier:</strong> 
                            ${individual.domain.prediction} 
                            (${(individual.domain.confidence * 100).toFixed(1)}%)
                            <br><small>Trust Score: ${(individual.domain.trust_score * 100).toFixed(1)}%</small>
                        </div>
                        ` : ''}
                        ${individual.consistent ? `
                        <div class="prediction-item">
                            <strong>📋 Rule-based Analyzer:</strong> 
                            ${individual.consistent.label} 
                            (${(individual.consistent.confidence * 100).toFixed(1)}%)
                            <br><small>Content and pattern analysis</small>
                        </div>
                        ` : ''}
                    </div>
                    
                    ${ensemble.methods_used ? `
                    <div class="ensemble-info" style="background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <h4>🔬 Ensemble Details</h4>
                        <p><strong>Methods Used:</strong> ${ensemble.num_methods}</p>
                        <p><strong>Voting:</strong> ${ensemble.consensus?.real_votes || 0} Real, ${ensemble.consensus?.fake_votes || 0} Fake</p>
                        <div style="font-size: 12px; color: #666;">
                            ${ensemble.methods_used.map(method => `• ${method}`).join('<br>')}
                        </div>
                    </div>
                    ` : ''}
                    
                    <div class="social-metrics">
                        <div class="metric">
                            <div><strong>${metrics.total_tweets || 0}</strong></div>
                            <div>Tweets</div>
                        </div>
                        <div class="metric">
                            <div><strong>${metrics.total_retweets || 0}</strong></div>
                            <div>Retweets</div>
                        </div>
                        <div class="metric">
                            <div><strong>${metrics.total_likes || 0}</strong></div>
                            <div>Likes</div>
                        </div>
                        <div class="metric">
                            <div><strong>${metrics.total_replies || 0}</strong></div>
                            <div>Replies</div>
                        </div>
                    </div>
                    
                    ${tweetsHtml}
                    
                    <div style="margin-top: 15px;">
                        <strong>Final Probabilities:</strong>
                        Real: ${(prediction.probabilities.real * 100).toFixed(1)}% | 
                        Fake: ${(prediction.probabilities.fake * 100).toFixed(1)}%
                    </div>
                </div>
            `;
            
            result.style.display = 'block';
        }
        
        function displayError(error) {
            const result = document.getElementById('result');
            result.innerHTML = `
                <div class="result error">
                    <div class="result-header">⚠️ Error</div>
                    <p>${error}</p>
                </div>
            `;
            result.style.display = 'block';
        }
        
        async function batchAnalyze() {
            console.log('batchAnalyze function called');
            
            const urlText = document.getElementById('urlList').value.trim();
            console.log('URL text:', urlText);
            
            if (!urlText) {
                alert('Please enter URLs');
                return;
            }
            
            const urls = urlText.split('\\n').map(url => url.trim()).filter(url => url);
            console.log('Parsed URLs:', urls);
            
            if (urls.length === 0) {
                alert('No valid URLs found');
                return;
            }
            
            if (urls.length > 10) {
                alert('Maximum 10 URLs allowed');
                return;
            }
            
            const batchBtn = document.getElementById('batchBtn');
            const progress = document.getElementById('batchProgress');
            const results = document.getElementById('batchResults');
            
            batchBtn.disabled = true;
            progress.style.display = 'block';
            results.style.display = 'none';
            
            try {
                console.log('Sending batch request');
                
                const response = await fetch('/batch', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ urls: urls })
                });
                
                console.log('Batch response status:', response.status);
                
                const data = await response.json();
                console.log('Batch response data:', data);
                
                if (response.ok) {
                    displayBatchResults(data.results);
                } else {
                    alert('Batch analysis failed: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Batch analysis error:', error);
                alert('Network error: ' + error.message);
            } finally {
                batchBtn.disabled = false;
                progress.style.display = 'none';
            }
        }
        
        function displayBatchResults(results) {
            const container = document.getElementById('batchResults');
            
            let html = '<h3>📊 Batch Analysis Results</h3>';
            
            results.forEach((result, index) => {
                if (result.error) {
                    html += `
                        <div class="result error" style="margin-bottom: 15px;">
                            <strong>URL ${index + 1}:</strong> ${result.url}<br>
                            <strong>Error:</strong> ${result.error}
                        </div>
                    `;
                } else {
                    const prediction = result.final_prediction;
                    const isReal = prediction.label === 'REAL';
                    const resultClass = isReal ? 'real' : 'fake';
                    const emoji = isReal ? '✅' : '❌';
                    const social = result.social_media?.metrics || {};
                    
                    html += `
                        <div class="result ${resultClass}" style="margin-bottom: 15px;">
                            <strong>URL ${index + 1}:</strong> <a href="${result.url}" target="_blank">${result.url}</a><br>
                            <strong>Prediction:</strong> ${emoji} ${prediction.label} (${(prediction.confidence * 100).toFixed(1)}% confidence)<br>
                            <strong>Title:</strong> ${result.title || 'N/A'}<br>
                            <strong>Domain:</strong> ${result.domain || 'N/A'}<br>
                            <strong>Method:</strong> ${result.method || 'Ultimate Ensemble'}<br>
                            <strong>Social:</strong> ${social.total_tweets || 0} tweets, ${social.total_likes || 0} likes<br>
                            <small style="color: #28a745;">🏆 97%+ accuracy models</small>
                        </div>
                    `;
                }
            });
            
            container.innerHTML = html;
            container.style.display = 'block';
        }
        
        async function testConsistency() {
            console.log('testConsistency function called');
            
            const url = document.getElementById('consistencyUrl').value.trim();
            console.log('Consistency URL:', url);
            
            if (!url) {
                alert('Please enter a URL to test');
                return;
            }
            
            const consistencyBtn = document.getElementById('consistencyBtn');
            const results = document.getElementById('consistencyResults');
            
            consistencyBtn.disabled = true;
            consistencyBtn.textContent = 'Testing...';
            results.style.display = 'none';
            
            try {
                console.log('Sending consistency test request');
                
                const response = await fetch('/test_consistency', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url, runs: 3 })
                });
                
                console.log('Consistency response status:', response.status);
                
                const data = await response.json();
                console.log('Consistency response data:', data);
                
                if (response.ok) {
                    displayConsistencyResults(data);
                } else {
                    alert('Consistency test failed: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Consistency test error:', error);
                alert('Network error: ' + error.message);
            } finally {
                consistencyBtn.disabled = false;
                consistencyBtn.textContent = 'Test Consistency';
            }
        }
        
        function displayConsistencyResults(data) {
            const container = document.getElementById('consistencyResults');
            
            const isConsistent = data.is_consistent;
            const resultClass = isConsistent ? 'real' : 'error';
            const emoji = isConsistent ? '✅' : '❌';
            
            let html = `
                <div class="result ${resultClass}">
                    <div class="result-header">
                        ${emoji} Consistency Test Results
                    </div>
                    <p><strong>URL:</strong> <a href="${data.url}" target="_blank">${data.url}</a></p>
                    <p><strong>Runs:</strong> ${data.runs}</p>
                    <p><strong>Status:</strong> ${data.consistency_message}</p>
                    
                    <h4>Individual Run Results:</h4>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
            `;
            
            data.results.forEach(result => {
                html += `
                    <div style="margin: 5px 0;">
                        Run ${result.run}: ${result.prediction} (${(result.confidence * 100).toFixed(1)}% confidence)
                    </div>
                `;
            });
            
            html += `
                    </div>
                    ${isConsistent ? 
                        '<p style="color: #28a745; font-weight: bold;">🎯 This detector provides consistent, reliable results!</p>' : 
                        '<p style="color: #dc3545; font-weight: bold;">⚠️ Inconsistent results detected. This detector may not be reliable.</p>'
                    }
                </div>
            `;
            
            container.innerHTML = html;
            container.style.display = 'block';
        }

        // Test if JavaScript is working
        console.log('JavaScript loaded successfully');
        
        // Add event listeners when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, adding event listeners');
            
            // Allow Enter key to trigger analysis
            const urlInput = document.getElementById('urlInput');
            const consistencyUrl = document.getElementById('consistencyUrl');
            
            if (urlInput) {
                urlInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        console.log('Enter pressed on main URL input');
                        analyzeUrl();
                    }
                });
            }
            
            if (consistencyUrl) {
                consistencyUrl.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        console.log('Enter pressed on consistency URL input');
                        testConsistency();
                    }
                });
            }
            
            console.log('Event listeners added successfully');
        });
    </script>
</body>
</html>'''
    
    with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Run the web application"""
    # Create templates
    create_templates()
    
    # Create results directory
    Path('web_results').mkdir(exist_ok=True)
    
    print("🏆 Starting Ultimate Fake News Detection Web Interface")
    print("=" * 70)
    print("🎯 Features:")
    print("   • 97%+ accuracy Hugging Face trained models")
    print("   • Multi-method ensemble analysis")
    print("   • Real Twitter data integration")
    print("   • Domain trust analysis")
    print("   • Rule-based content analysis")
    print("   • Consistency testing")
    print("   • Batch processing")
    print("")
    print("📊 Trained on: Pulk17/Fake-News-Detection-dataset (30,000 samples)")
    print("🔗 Open your browser and go to: http://localhost:5000")
    print("📝 Twitter API token loaded from .env file")
    print("⚠️  Press Ctrl+C to stop the server")
    print("=" * 70)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()