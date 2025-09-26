#!/usr/bin/env python3

"""
Debug version of web interface to test what's going wrong
"""

from flask import Flask, request, jsonify
from fixed_improved_detector import FixedImprovedDetector
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize detector
detector = FixedImprovedDetector()
logger.info("Detector initialized")

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Debug Fake News Detector</title>
</head>
<body>
    <h1>Debug Fake News Detector</h1>
    <input type="url" id="urlInput" placeholder="Enter URL..." style="width: 500px; padding: 10px;">
    <button onclick="analyze()" style="padding: 10px;">Analyze</button>
    <div id="result" style="margin-top: 20px; padding: 20px; border: 1px solid #ccc;"></div>
    
    <script>
        async function analyze() {
            const url = document.getElementById('urlInput').value;
            console.log('Analyzing URL:', url);
            
            if (!url) {
                alert('Please enter a URL');
                return;
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                console.log('Response:', data);
                
                document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = 'Error: ' + error.message;
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')
        
        logger.info(f"Received request to analyze: {url}")
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Test the detector
        result = detector.predict(url)
        logger.info(f"Detector result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🐛 Debug Web Interface")
    print("Open: http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)