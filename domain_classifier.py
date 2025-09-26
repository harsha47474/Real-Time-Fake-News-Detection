#!/usr/bin/env python3

"""
Domain-based News Classifier
Simple but effective classifier based on domain reputation
"""

import argparse
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

class DomainClassifier:
    def __init__(self):
        # Trusted news domains (high credibility)
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
            'apnews.com': 0.95
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            'breaking', 'shocking', 'unbelievable', 'exclusive', 
            'leaked', 'secret', 'hidden truth', 'they dont want you to know'
        ]
    
    def classify(self, url):
        """Classify URL based on domain and content analysis"""
        domain = urlparse(url).netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check trusted domains
        trust_score = 0.5  # Default neutral
        for trusted_domain, score in self.trusted_domains.items():
            if trusted_domain in domain:
                trust_score = score
                break
        
        # Get content for additional analysis
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            # Check for suspicious patterns
            suspicious_count = sum(1 for pattern in self.suspicious_patterns 
                                 if pattern.lower() in title_text.lower())
            
            # Adjust score based on content
            if suspicious_count > 0:
                trust_score -= 0.1 * suspicious_count
            
            # HTTPS bonus
            if url.startswith('https'):
                trust_score += 0.05
                
        except:
            title_text = "Could not fetch title"
        
        # Final classification
        trust_score = max(0.0, min(1.0, trust_score))  # Clamp to [0,1]
        
        if trust_score >= 0.7:
            prediction = "REAL"
            confidence = trust_score
        else:
            prediction = "FAKE"
            confidence = 1 - trust_score
        
        return {
            'url': url,
            'domain': domain,
            'title': title_text,
            'prediction': prediction,
            'confidence': confidence,
            'trust_score': trust_score,
            'method': 'Domain-based Classification'
        }

def main():
    parser = argparse.ArgumentParser(description='Domain-based News Classifier')
    parser.add_argument('--url', type=str, required=True, help='URL to classify')
    
    args = parser.parse_args()
    
    print("🌐 Domain-based News Classifier")
    print("=" * 40)
    
    classifier = DomainClassifier()
    result = classifier.classify(args.url)
    
    print(f"🔗 URL: {result['url']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"📰 Title: {result['title']}")
    print(f"🔍 Method: {result['method']}")
    
    emoji = "✅" if result['prediction'] == "REAL" else "❌"
    print(f"\n{emoji} PREDICTION: {result['prediction']}")
    print(f"🎯 Confidence: {result['confidence']:.1%}")
    print(f"📊 Domain Trust Score: {result['trust_score']:.1%}")

if __name__ == '__main__':
    main()