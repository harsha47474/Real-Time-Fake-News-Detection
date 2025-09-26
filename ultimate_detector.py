#!/usr/bin/env python3

"""
Ultimate Fake News Detector - Ensembles ALL detectors for maximum accuracy
Uses Hugging Face trained models (97%+ accuracy) + All other detectors + Keyword detection
"""

import argparse
from huggingface_detector import HuggingFaceDetector
from domain_classifier import DomainClassifier
from consistent_detector import ConsistentFakeNewsDetector
try:
    from test_fast_models import FastModelDetector
    FAST_MODELS_AVAILABLE = True
except ImportError:
    FAST_MODELS_AVAILABLE = False
    FastModelDetector = None
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class UltimateDetector:
    """Ultimate detector combining ALL available methods"""
    
    def __init__(self):
        print("🚀 Initializing Ultimate Fake News Detector...")
        print("🔄 Loading ALL available detectors...")
        
        # Initialize all detectors
        self.hf_detector = HuggingFaceDetector()
        self.domain_classifier = DomainClassifier()
        self.consistent_detector = ConsistentFakeNewsDetector()
        # Initialize fast detector if available
        if FAST_MODELS_AVAILABLE:
            try:
                self.fast_detector = FastModelDetector()
                self.has_fast_models = (self.fast_detector.sklearn_models is not None)
            except Exception as e:
                print(f"⚠️  Fast models not available: {e}")
                self.fast_detector = None
                self.has_fast_models = False
        else:
            self.fast_detector = None
            self.has_fast_models = False
        
        # Check availability
        self.has_hf_models = (self.hf_detector.models is not None)
        
        # Fake content keywords (if URL contains these, it's likely fake)
        self.fake_keywords = [
            'fake', 'hoax', 'scam', 'conspiracy', 'debunked', 'false', 'misleading',
            'clickbait', 'satire', 'parody', 'rumor', 'unverified', 'alleged',
            'breaking', 'shocking', 'unbelievable', 'exclusive', 'leaked',
            'secret', 'hidden truth', 'they dont want you to know',
            'doctors hate this', 'miracle cure', 'government cover-up'
        ]
        
        print(f"✅ Hugging Face Models (97%+): {'Available' if self.has_hf_models else 'Not Available'}")
        print(f"✅ Fast ML Models: {'Available' if self.has_fast_models else 'Not Available'}")
        print("✅ Domain Classifier: Available")
        print("✅ Rule-based Detector: Available")
        print("✅ Keyword Detection: Available")
        
        if self.has_hf_models:
            print("🎯 Using state-of-the-art 97%+ accuracy models!")
        
        total_methods = sum([
            self.has_hf_models,
            self.has_fast_models,
            True,  # Domain classifier
            True,  # Rule-based
            True   # Keyword detection
        ])
        print(f"🔥 Total methods available: {total_methods}")
    
    def extract_content_for_analysis(self, url):
        """Extract content for keyword analysis"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title and content
            title = soup.find('title')
            title_text = title.get_text() if title else ""
            
            # Extract all text content
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            content = soup.get_text()
            full_text = f"{title_text} {content}".lower()
            
            return {
                'title': title_text,
                'content': content[:2000],  # Limit for analysis
                'full_text': full_text,
                'url_text': url.lower()
            }
        except Exception as e:
            print(f"⚠️  Error extracting content: {e}")
            return None
    
    def keyword_detection(self, url, content_data):
        """Detect fake news based on keywords in URL and content"""
        if not content_data:
            return None
        
        fake_score = 0
        detected_keywords = []
        
        # Check URL for fake keywords (URL keywords are very strong indicators)
        url_text = content_data['url_text']
        for keyword in self.fake_keywords:
            if keyword in url_text:
                if keyword in ['fake', 'hoax', 'scam', 'conspiracy']:
                    fake_score += 5  # Very strong URL indicators
                else:
                    fake_score += 3  # Strong URL indicators
                detected_keywords.append(f"URL: {keyword}")
        
        # Check title and content for fake keywords
        full_text = content_data['full_text']
        for keyword in self.fake_keywords:
            if keyword in full_text:
                fake_score += 1
                detected_keywords.append(f"Content: {keyword}")
        
        # Special case: if "fake" is explicitly in the content
        if 'fake' in full_text:
            fake_score += 5  # Very strong indicator
            detected_keywords.append("EXPLICIT: 'fake' found in content")
        
        # Calculate confidence based on keyword count
        if fake_score >= 5:  # Very high fake score (strong URL keywords)
            prediction = 'FAKE'
            confidence = 0.9
        elif fake_score >= 3:  # High fake score
            prediction = 'FAKE'
            confidence = 0.8
        elif fake_score > 0:  # Some fake indicators
            prediction = 'FAKE'
            confidence = 0.7
        else:  # No fake indicators
            prediction = 'REAL'
            confidence = 0.6  # Moderate confidence for no keywords
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'real': 1 - confidence if prediction == 'FAKE' else confidence,
                'fake': confidence if prediction == 'FAKE' else 1 - confidence
            },
            'fake_score': fake_score,
            'detected_keywords': detected_keywords[:5]  # Limit to top 5
        }
    
    def predict(self, url):
        """Ultimate prediction using ALL available methods"""
        print(f"🔍 Ultimate analysis of: {url}")
        
        # First, extract content for keyword analysis
        content_data = self.extract_content_for_analysis(url)
        
        results = {}
        
        # Keyword detection (highest priority if "fake" found explicitly)
        keyword_result = self.keyword_detection(url, content_data)
        if keyword_result:
            results['keyword'] = keyword_result
            # If explicit "fake" found, return immediately
            if any('EXPLICIT' in kw for kw in keyword_result.get('detected_keywords', [])):
                print("🚨 EXPLICIT 'fake' content detected - returning FAKE immediately")
                return {
                    'url': url,
                    'title': content_data['title'] if content_data else 'Unknown',
                    'domain': urlparse(url).netloc.replace('www.', ''),
                    'final_prediction': {
                        'label': 'FAKE',
                        'confidence': 0.95,
                        'probabilities': {'real': 0.05, 'fake': 0.95}
                    },
                    'individual_predictions': {
                        'keyword': keyword_result,
                        'reason': 'Explicit fake content detected'
                    },
                    'ensemble_info': {
                        'methods_used': ['Keywords (Explicit Fake Content)'],
                        'weights': {'Keywords': 1.0},
                        'num_methods': 1,
                        'consensus': {'real_votes': 0, 'fake_votes': 1}
                    },
                    'method': 'Ultimate Ensemble (Explicit Fake Content Detected)',
                    'detected_keywords': keyword_result['detected_keywords']
                }
        
        # Get predictions from all methods
        try:
            if self.has_hf_models:
                hf_result = self.hf_detector.predict(url)
                # Check if HF detector returned an error
                if 'error' in hf_result:
                    print(f"❌ Cannot analyze URL: {hf_result['error']}")
                    return {
                        'url': url,
                        'title': 'Error',
                        'domain': 'unknown',
                        'final_prediction': {
                            'label': 'ERROR',
                            'confidence': 0.0,
                            'probabilities': {'real': 0.0, 'fake': 0.0}
                        },
                        'method': 'Error - Invalid URL or no content',
                        'error': hf_result['error']
                    }
                results['huggingface'] = hf_result
        except Exception as e:
            print(f"⚠️  Hugging Face models failed: {e}")
            results['huggingface'] = None
        
        # Fast models
        try:
            if self.has_fast_models:
                fast_result = self.fast_detector.predict(url)
                results['fast_models'] = fast_result
        except Exception as e:
            print(f"⚠️  Fast models failed: {e}")
            results['fast_models'] = None
        
        try:
            domain_result = self.domain_classifier.classify(url)
            results['domain'] = domain_result
        except Exception as e:
            print(f"⚠️  Domain classification failed: {e}")
            results['domain'] = None
        
        try:
            consistent_result = self.consistent_detector.predict(url)
            results['consistent'] = consistent_result
        except Exception as e:
            print(f"⚠️  Consistent prediction failed: {e}")
            results['consistent'] = None
        
        # Combine all predictions with intelligent weighting
        return self.ultimate_ensemble(url, results)
    
    def ultimate_ensemble(self, url, results):
        """Ultimate ensemble with intelligent weighting based on model performance"""
        predictions = []
        weights = []
        methods = []
        
        # Keyword detection (highest priority for explicit fake content)
        if results.get('keyword'):
            keyword_pred = results['keyword']
            keyword_real_prob = keyword_pred['probabilities']['real']
            
            # Much higher weight if fake keywords detected, especially in URL
            if keyword_pred['fake_score'] > 5:
                weight = 0.6  # Very high fake score
            elif keyword_pred['fake_score'] > 2:
                weight = 0.5  # High fake score (especially URL keywords)
            elif keyword_pred['fake_score'] > 0:
                weight = 0.3  # Some fake indicators
            else:
                weight = 0.1  # No fake indicators
            
            predictions.append(keyword_real_prob)
            weights.append(weight)
            methods.append(f"Keywords ({keyword_pred['confidence']:.1%})")
        
        # Hugging Face models (highest ML accuracy - 97%+)
        if results.get('huggingface') and self.has_hf_models:
            hf_pred = results['huggingface']['prediction']
            hf_real_prob = hf_pred['probabilities']['real']
            confidence = hf_pred['confidence']
            
            # High weight for high-accuracy models
            if confidence > 0.9:
                weight = 0.5  # Very confident
            elif confidence > 0.8:
                weight = 0.4  # Confident
            else:
                weight = 0.3  # Less confident but still high accuracy
            
            predictions.append(hf_real_prob)
            weights.append(weight)
            methods.append(f"HF Models ({confidence:.1%})")
        
        # Fast ML models
        if results.get('fast_models') and self.has_fast_models:
            fast_pred = results['fast_models']['prediction']
            fast_real_prob = fast_pred['probabilities']['real']
            confidence = fast_pred['confidence']
            
            predictions.append(fast_real_prob)
            weights.append(0.2)  # Medium weight
            methods.append(f"Fast ML ({confidence:.1%})")
        
        # Domain classifier (reliable for known domains)
        if results.get('domain'):
            domain_pred = results['domain']
            trust_score = domain_pred['trust_score']
            
            if domain_pred['prediction'] == 'REAL':
                domain_real_prob = domain_pred['confidence']
            else:
                domain_real_prob = 1 - domain_pred['confidence']
            
            # Weight based on domain trust
            if trust_score > 0.9:
                weight = 0.25  # Very trusted domain
            elif trust_score > 0.7:
                weight = 0.2   # Trusted domain
            else:
                weight = 0.1   # Unknown domain
            
            predictions.append(domain_real_prob)
            weights.append(weight)
            methods.append(f"Domain ({domain_pred['confidence']:.1%})")
        
        # Rule-based detector (consistent backup)
        if results.get('consistent'):
            consistent_pred = results['consistent']['prediction']
            consistent_real_prob = consistent_pred['probabilities']['real']
            
            predictions.append(consistent_real_prob)
            weights.append(0.1)  # Lower weight for rule-based
            methods.append(f"Rules ({consistent_pred['confidence']:.1%})")
        
        # Calculate intelligent weighted ensemble
        if predictions:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average
            final_real_prob = sum(p * w for p, w in zip(predictions, normalized_weights))
            final_fake_prob = 1 - final_real_prob
            
            # Apply consensus boost
            consensus_threshold = 0.7
            real_consensus = sum(1 for p in predictions if p > consensus_threshold)
            fake_consensus = sum(1 for p in predictions if p < (1 - consensus_threshold))
            
            if real_consensus >= 2:  # At least 2 methods agree it's real
                final_real_prob = min(0.95, final_real_prob + 0.05)
            elif fake_consensus >= 2:  # At least 2 methods agree it's fake
                final_real_prob = max(0.05, final_real_prob - 0.05)
            
            final_fake_prob = 1 - final_real_prob
            
            # Final prediction
            if final_real_prob > final_fake_prob:
                final_label = 'REAL'
                confidence = final_real_prob
            else:
                final_label = 'FAKE'
                confidence = final_fake_prob
        else:
            # Fallback
            final_label = 'UNKNOWN'
            confidence = 0.5
            final_real_prob = 0.5
            final_fake_prob = 0.5
        
        # Get basic info
        title = "Unknown"
        domain = "unknown"
        
        for result_key in ['huggingface', 'domain', 'consistent']:
            if results[result_key]:
                if 'title' in results[result_key]:
                    title = results[result_key]['title']
                if 'domain' in results[result_key]:
                    domain = results[result_key]['domain']
                break
        
        # Prepare individual predictions for output
        individual_preds = {}
        
        if results.get('keyword'):
            individual_preds['keyword'] = results['keyword']
        
        if results.get('huggingface'):
            individual_preds['huggingface'] = results['huggingface']['prediction']
        
        if results.get('fast_models'):
            individual_preds['fast_models'] = results['fast_models']['prediction']
        
        if results.get('domain'):
            individual_preds['domain'] = {
                'prediction': results['domain']['prediction'],
                'confidence': results['domain']['confidence'],
                'trust_score': results['domain']['trust_score']
            }
        
        if results.get('consistent'):
            individual_preds['consistent'] = results['consistent']['prediction']
        
        return {
            'url': url,
            'title': title,
            'domain': domain,
            'final_prediction': {
                'label': final_label,
                'confidence': confidence,
                'probabilities': {
                    'real': final_real_prob,
                    'fake': final_fake_prob
                }
            },
            'individual_predictions': individual_preds,
            'ensemble_info': {
                'methods_used': methods,
                'weights': dict(zip(['Keywords', 'HF', 'Fast', 'Domain', 'Rules'], 
                               normalized_weights[:len(methods)])) if predictions else {},
                'num_methods': len(predictions),
                'consensus': {
                    'real_votes': sum(1 for p in predictions if p > 0.5),
                    'fake_votes': sum(1 for p in predictions if p < 0.5)
                } if predictions else {'real_votes': 0, 'fake_votes': 0}
            },
            'detected_keywords': results.get('keyword', {}).get('detected_keywords', []),
            'method': f'Ultimate Ensemble ({len(predictions)} methods: Keywords + HF 97%+ + Fast ML + Domain + Rules)'
        }
    
    def test_fake_detection(self):
        """Built-in test method for fake detection capabilities"""
        print("🧪 Testing Ultimate Detector Fake Detection")
        print("=" * 60)
        
        # Test cases with different types of content
        test_cases = [
            {
                'url': 'https://www.reuters.com/world/',
                'expected': 'REAL',
                'description': 'Trusted Reuters source'
            },
            {
                'url': 'https://www.bbc.com/news',
                'expected': 'REAL',
                'description': 'Trusted BBC source'
            },
            {
                'url': 'https://timesofindia.indiatimes.com/business/india-business/trumps-100-tariffs-on-pharma-sun-pharma-biocon-cipla-other-pharmaceutical-stocks-tank-jitters-on-d-street/articleshow/124144816.cms',
                'expected': 'REAL',
                'description': 'Times of India business article'
            },
            {
                'url': 'https://example.com/fake-news-breaking-story',
                'expected': 'FAKE',
                'description': 'URL with "fake" keyword'
            },
            {
                'url': 'https://conspiracy-news.com/shocking-truth',
                'expected': 'FAKE',
                'description': 'URL with "conspiracy" keyword'
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case['description']}")
            print(f"URL: {test_case['url']}")
            print(f"Expected: {test_case['expected']}")
            print("-" * 40)
            
            try:
                # Test keyword detection first
                content_data = self.extract_content_for_analysis(test_case['url'])
                if content_data:
                    keyword_result = self.keyword_detection(test_case['url'], content_data)
                    if keyword_result and keyword_result['detected_keywords']:
                        print(f"🔍 Keywords found: {', '.join(keyword_result['detected_keywords'][:3])}")
                        print(f"🚨 Fake score: {keyword_result['fake_score']}")
                
                # Full prediction
                result = self.predict(test_case['url'])
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    results.append({'test': i, 'status': 'ERROR'})
                    continue
                
                prediction = result['final_prediction']['label']
                confidence = result['final_prediction']['confidence']
                
                # Check if prediction matches expected
                is_correct = prediction == test_case['expected']
                status_emoji = "✅" if is_correct else "❌"
                
                print(f"{status_emoji} Result: {prediction} ({confidence:.1%})")
                print(f"📊 Methods: {result['ensemble_info']['num_methods']}")
                print(f"🗳️  Votes: {result['ensemble_info']['consensus']['real_votes']} Real, {result['ensemble_info']['consensus']['fake_votes']} Fake")
                
                # Show individual predictions briefly
                individual = result['individual_predictions']
                for method, pred in individual.items():
                    if isinstance(pred, dict):
                        if 'prediction' in pred:
                            print(f"   • {method}: {pred['prediction']}")
                        elif 'label' in pred:
                            print(f"   • {method}: {pred['label']}")
                
                results.append({
                    'test': i,
                    'status': 'CORRECT' if is_correct else 'INCORRECT',
                    'prediction': prediction,
                    'expected': test_case['expected'],
                    'confidence': confidence
                })
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results.append({'test': i, 'status': 'EXCEPTION'})
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        
        correct = sum(1 for r in results if r['status'] == 'CORRECT')
        total = len([r for r in results if r['status'] in ['CORRECT', 'INCORRECT']])
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"✅ Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"🎯 Available methods: {sum([self.has_hf_models, self.has_fast_models, True, True, True])}")
        print(f"🔥 Keyword detection: {'Active' if any('fake' in url for url in [tc['url'] for tc in test_cases]) else 'Tested'}")
        
        for result in results:
            status_emoji = "✅" if result['status'] == 'CORRECT' else "❌" if result['status'] == 'INCORRECT' else "⚠️"
            print(f"{status_emoji} Test {result['test']}: {result['status']}")
        
        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Ultimate Fake News Detector')
    parser.add_argument('--url', type=str, help='URL to analyze')
    parser.add_argument('--test', action='store_true', help='Run built-in fake detection tests')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = UltimateDetector()
    
    # Run tests if requested
    if args.test:
        print("🏆 Ultimate Fake News Detector - Test Mode")
        print("=" * 60)
        accuracy = detector.test_fake_detection()
        print(f"\n🎉 Testing completed with {accuracy:.1f}% accuracy!")
        return
    
    # Require URL if not testing
    if not args.url:
        print("❌ Please provide a URL to analyze or use --test to run tests")
        print("💡 Examples:")
        print("   python ultimate_detector.py --url \"https://www.reuters.com/world/\"")
        print("   python ultimate_detector.py --test")
        return
    
    print("🏆 Ultimate Fake News Detector")
    print("=" * 60)
    print("🎯 Maximum Accuracy: Hugging Face (97%+) + All Methods + Keyword Detection")
    print("📊 Trained on: Pulk17/Fake-News-Detection-dataset")
    
    # Validate URL
    url = args.url
    if not url.startswith(('http://', 'https://')):
        print(f"\n❌ Invalid URL: {url}")
        print("💡 Please provide a valid URL starting with http:// or https://")
        print("📝 Example: python ultimate_detector.py --url \"https://www.reuters.com/world/\"")
        return
    
    if url == "your_url_here":
        print(f"\n❌ Please replace 'your_url_here' with an actual URL")
        print("📝 Example: python ultimate_detector.py --url \"https://www.reuters.com/world/\"")
        return
    
    # Analyze URL
    result = detector.predict(url)
    
    # Check for errors
    if 'error' in result:
        print(f"\n❌ ANALYSIS ERROR")
        print("=" * 30)
        print(f"🔗 URL: {result['url']}")
        print(f"⚠️  Error: {result['error']}")
        print(f"💡 Please check the URL and try again")
        return
    
    # Display results
    print(f"\n📊 ULTIMATE ANALYSIS RESULTS")
    print("=" * 40)
    print(f"🔗 URL: {result['url']}")
    print(f"📰 Title: {result['title']}")
    print(f"🌐 Domain: {result['domain']}")
    print(f"🔍 Method: {result['method']}")
    
    # Final prediction
    final = result['final_prediction']
    emoji = "✅" if final['label'] == "REAL" else "❌" if final['label'] == "FAKE" else "❓"
    print(f"\n{emoji} FINAL PREDICTION: {final['label']}")
    print(f"🎯 Confidence: {final['confidence']:.1%}")
    print(f"📊 Probabilities:")
    print(f"   Real: {final['probabilities']['real']:.1%}")
    print(f"   Fake: {final['probabilities']['fake']:.1%}")
    
    # Ensemble info
    if 'ensemble_info' in result:
        ensemble = result['ensemble_info']
        consensus = ensemble['consensus']
        print(f"\n🔬 ENSEMBLE DETAILS:")
        print(f"📈 Methods used: {ensemble['num_methods']}")
        print(f"🗳️  Voting: {consensus['real_votes']} Real, {consensus['fake_votes']} Fake")
        for method in ensemble['methods_used']:
            print(f"   • {method}")
        
        final_message = f"🏆 Ultimate analysis completed with maximum accuracy using {ensemble['num_methods']} methods!"
    else:
        final_message = "🏆 Ultimate analysis completed!"
    
    # Individual predictions
    individual = result['individual_predictions']
    print(f"\n🔍 INDIVIDUAL PREDICTIONS:")
    
    if individual.get('keyword'):
        keyword = individual['keyword']
        print(f"🔍 Keyword Detection: {keyword['prediction']} ({keyword['confidence']:.1%})")
        if keyword.get('detected_keywords'):
            print(f"   Keywords found: {', '.join(keyword['detected_keywords'][:3])}")
    
    if individual.get('huggingface'):
        hf = individual['huggingface']
        print(f"🤖 Hugging Face (97%+): {hf['label']} ({hf['confidence']:.1%})")
    
    if individual.get('fast_models'):
        fast = individual['fast_models']
        print(f"⚡ Fast ML Models: {fast['label']} ({fast['confidence']:.1%})")
    
    if individual.get('domain'):
        domain = individual['domain']
        print(f"🌐 Domain: {domain['prediction']} ({domain['confidence']:.1%})")
    
    if individual.get('consistent'):
        consistent = individual['consistent']
        print(f"📋 Rules: {consistent['label']} ({consistent['confidence']:.1%})")
    
    # Show detected keywords if any
    if result.get('detected_keywords'):
        print(f"\n🚨 DETECTED KEYWORDS:")
        for keyword in result['detected_keywords'][:5]:
            print(f"   • {keyword}")
    
    print(f"\n{final_message}")

if __name__ == '__main__':
    main()