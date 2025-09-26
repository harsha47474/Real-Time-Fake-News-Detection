#!/usr/bin/env python3

"""
Simple Ultimate Web Interface - Handles errors gracefully
"""

from flask import Flask, request, jsonify
from ultimate_detector import UltimateDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize detector
detector = None

def init_detector():
    global detector
    if detector is None:
        try:
            detector = UltimateDetector()
            logger.info("Ultimate detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            detector = None

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Ultimate Fake News Detector (97%+ Accuracy)</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .input-section { margin-bottom: 20px; }
        .url-input { width: 70%; padding: 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
        .analyze-btn { width: 25%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-left: 10px; }
        .analyze-btn:hover { background: #0056b3; }
        .analyze-btn:disabled { background: #ccc; cursor: not-allowed; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; border-left: 5px solid; }
        .result.real { background: #f0fff4; border-color: #38a169; }
        .result.fake { background: #fff5f5; border-color: #e53e3e; }
        .result.error { background: #fffbf0; border-color: #d69e2e; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .individual-predictions { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .prediction-item { margin: 10px 0; padding: 8px; background: white; border-radius: 3px; border-left: 3px solid #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Ultimate Fake News Detector</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            🎯 97%+ Accuracy | 🤖 Hugging Face Trained | 🔍 Multi-Method Analysis
        </p>
        
        <div class="input-section">
            <input type="url" id="urlInput" class="url-input" placeholder="Enter news article URL..." />
            <button onclick="analyzeUrl()" class="analyze-btn" id="analyzeBtn">Analyze</button>
        </div>
        
        <div id="loading" class="loading" style="display: none;">
            🔍 Analyzing with 97%+ accuracy models... Please wait.
        </div>
        
        <div id="result" style="display: none;"></div>
    </div>

    <script>
        async function analyzeUrl() {
            const url = document.getElementById('urlInput').value.trim();
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
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    displayError(data.error);
                } else {
                    displayResult(data);
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResult(data) {
            const result = document.getElementById('result');
            const prediction = data.final_prediction;
            const individual = data.individual_predictions || {};
            
            const isReal = prediction.label === 'REAL';
            const resultClass = isReal ? 'real' : 'fake';
            const emoji = isReal ? '✅' : '❌';
            
            let individualHtml = '';
            if (individual.huggingface) {
                individualHtml += `<div class="prediction-item">
                    <strong>🤖 Hugging Face Models (97%+):</strong> ${individual.huggingface.label} (${(individual.huggingface.confidence * 100).toFixed(1)}%)
                </div>`;
            }
            if (individual.domain) {
                individualHtml += `<div class="prediction-item">
                    <strong>🌐 Domain Classifier:</strong> ${individual.domain.prediction} (${(individual.domain.confidence * 100).toFixed(1)}%)
                </div>`;
            }
            if (individual.consistent) {
                individualHtml += `<div class="prediction-item">
                    <strong>📋 Rule-based:</strong> ${individual.consistent.label} (${(individual.consistent.confidence * 100).toFixed(1)}%)
                </div>`;
            }
            
            result.innerHTML = `
                <div class="result ${resultClass}">
                    <h3>${emoji} ${prediction.label}</h3>
                    <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Title:</strong> ${data.title}</p>
                    <p><strong>Domain:</strong> ${data.domain}</p>
                    <p><strong>Method:</strong> ${data.method}</p>
                    <p><strong>URL:</strong> <a href="${data.url}" target="_blank">${data.url}</a></p>
                    
                    ${individualHtml ? `<div class="individual-predictions">
                        <h4>🔍 Individual Model Results</h4>
                        ${individualHtml}
                    </div>` : ''}
                    
                    <p><strong>Probabilities:</strong> Real: ${(prediction.probabilities.real * 100).toFixed(1)}%, Fake: ${(prediction.probabilities.fake * 100).toFixed(1)}%</p>
                </div>
            `;
            
            result.style.display = 'block';
        }
        
        function displayError(error) {
            const result = document.getElementById('result');
            result.innerHTML = `
                <div class="result error">
                    <h3>⚠️ Error</h3>
                    <p>${error}</p>
                    <p>Please check the URL and try again.</p>
                </div>
            `;
            result.style.display = 'block';
        }
        
        document.getElementById('urlInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') analyzeUrl();
        });
    </script>
</body>
</html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Initialize detector if needed
        init_detector()
        
        if detector is None:
            return jsonify({'error': 'Detector not available'}), 500
        
        # Analyze the URL
        result = detector.predict(url)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏆 Simple Ultimate Fake News Detector Web Interface")
    print("=" * 60)
    print("🎯 97%+ Accuracy | 🤖 Hugging Face Trained")
    print("🔗 Open: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=False)