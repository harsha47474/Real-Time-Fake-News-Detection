#!/usr/bin/env python3

"""
Consistent Fake News Detector - Gives same results every time
"""

import argparse
import hashlib
import numpy as np
from domain_classifier import DomainClassifier
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

class ConsistentFakeNewsDetector:
    def __init__(self):
        self.domain_classifier = DomainClassifier()
        
        # Trusted domains with their credibility scores
        self.trusted_domains = {
            'reuters.com': 0.95,
            'bbc.com': 0.95,
            'cnn.com': 0.90,
            'nytimes.com': 0.95,
            'washingtonpost.com': 0.90,
            'thehindu.com': 0.90,
            'timesofindia.indiatimes.com': 0.85,
            'ndtv.com': 0.85,
            'indianexpress.com': 0.85,
            'theguardian.com': 0.90,
            'npr.org': 0.90,
            'apnews.com': 0.95,
            'abcnews.go.com': 0.85,
            'cbsnews.com': 0.85,
            'nbcnews.com': 0.85
        }
        
        # Suspicious domains (known for misinformation)
        self.suspicious_domains = {
            'infowars.com': 0.1,
            'breitbart.com': 0.3,
            'naturalnews.com': 0.2,
            'beforeitsnews.com': 0.1
        }
        
        # Suspicious keywords that often appear in fake news
        self.suspicious_keywords = [
            'breaking', 'shocking', 'unbelievable', 'exclusive', 'leaked',
            'secret', 'hidden truth', 'they dont want you to know',
            'doctors hate this', 'miracle cure', 'government cover-up'
        ]
    
    def get_domain_score(self, domain):
        """Get consistent domain credibility score"""
        domain = domain.lower().replace('www.', '')
        
        # Check trusted domains
        for trusted_domain, score in self.trusted_domains.items():
            if trusted_domain in domain:
                return score
        
        # Check suspicious domains
        for suspicious_domain, score in self.suspicious_domains.items():
            if suspicious_domain in domain:
                return score
        
        # Default score for unknown domains
        return 0.5
    
    def analyze_content(self, title, content):
        """Analyze content for suspicious patterns"""
        text = f"{title} {content}".lower()
        
        # Count suspicious keywords
        suspicious_count = sum(1 for keyword in self.suspicious_keywords 
                             if keyword in text)
        
        # Analyze title characteristics
        title_words = title.split()
        title_caps = sum(1 for word in title_words if word.isupper())
        title_caps_ratio = title_caps / len(title_words) if title_words else 0
        
        # Calculate content quality score
        content_score = 0.7  # Base score
        
        # Penalize excessive suspicious keywords
        if suspicious_count > 2:
            content_score -= 0.2
        elif suspicious_count > 0:
            content_score -= 0.1
        
        # Penalize excessive capitalization
        if title_caps_ratio > 0.5:
            content_score -= 0.15
        
        # Bonus for longer, detailed content
        if len(content.split()) > 200:
            content_score += 0.1
        
        return max(0.0, min(1.0, content_score))
    
    def get_url_features(self, url):
        """Extract consistent features from URL"""
        # Create deterministic hash from URL for consistent "randomness"
        url_hash = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
        np.random.seed(url_hash % 1000)  # Consistent seed based on URL
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = self._get_title(soup)
            content = self._get_content(soup)
            
        except Exception as e:
            title = "Could not fetch title"
            content = "Could not fetch content"
        
        domain = urlparse(url).netloc.replace('www.', '')
        
        return title, content, domain
    
    def _get_title(self, soup):
        """Extract title from HTML"""
        for selector in ['h1', 'title', '[property="og:title"]']:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "Unknown Title"
    
    def _get_content(self, soup):
        """Extract content from HTML"""
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        return content[:1000] if content else "No content found"
    
    def predict(self, url):
        """Make consistent prediction for URL"""
        print(f"🔍 Analyzing: {url}")
        
        # Get URL features
        title, content, domain = self.get_url_features(url)
        
        # Get domain credibility score
        domain_score = self.get_domain_score(domain)
        
        # Analyze content quality
        content_score = self.analyze_content(title, content)
        
        # Technical factors
        https_bonus = 0.05 if url.startswith('https') else 0
        
        # Calculate final score (weighted combination)
        final_score = (
            domain_score * 0.6 +      # Domain trust is most important
            content_score * 0.3 +     # Content quality
            https_bonus               # Technical factors
        )
        
        # Determine prediction
        if final_score >= 0.6:
            prediction = 'REAL'
            confidence = final_score
        else:
            prediction = 'FAKE'
            confidence = 1 - final_score
        
        return {
            'url': url,
            'title': title,
            'domain': domain,
            'prediction': {
                'label': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': final_score,
                    'fake': 1 - final_score
                }
            },
            'analysis': {
                'domain_score': domain_score,
                'content_score': content_score,
                'https_bonus': https_bonus,
                'final_score': final_score
            },
            'method': 'Consistent Rule-Based Analysis'
        }

def main():
    parser = argparse.ArgumentParser(description='Consistent Fake News Detector')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    
    print("🎯 Consistent Fake News Detector")
    print("=" * 50)
    print("✅ Same URL will always give the same result!")
    
    detector = ConsistentFakeNewsDetector()
    
    # Test multiple times to show consistency
    print(f"\n🔄 Testing consistency with 3 runs:")
    
    for i in range(3):
        result = detector.predict(args.url)
        print(f"Run {i+1}: {result['prediction']['label']} ({result['prediction']['confidence']:.1%})")
    
    # Show detailed results
    result = detector.predict(args.url)
    
    print(f"\n📊 DETAILED ANALYSIS")
    print("=" * 30)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    prediction = result['prediction']
    emoji = "✅" if prediction['label'] == "REAL" else "❌"
    print(f"\n{emoji} PREDICTION: {prediction['label']}")
    print(f"🎯 Confidence: {prediction['confidence']:.1%}")
    
    analysis = result['analysis']
    print(f"\n📈 SCORE BREAKDOWN:")
    print(f"   🌐 Domain Score: {analysis['domain_score']:.1%}")
    print(f"   📝 Content Score: {analysis['content_score']:.1%}")
    print(f"   🔒 HTTPS Bonus: {analysis['https_bonus']:.1%}")
    print(f"   📊 Final Score: {analysis['final_score']:.1%}")
    
    print(f"\n💡 This detector gives consistent results every time!")

if __name__ == '__main__':
    main()