"""
Text Humanizer API Server
Flask backend that combines NLP techniques with Cerebras AI for text humanization.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

from humanizer import humanize_text
from cerebras_client import humanize_with_ai, polish_with_ai

app = Flask(__name__, static_folder='.')
CORS(app)


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS)."""
    return send_from_directory('.', filename)


@app.route('/api/humanize', methods=['POST'])
def humanize():
    """
    Main humanization endpoint.
    
    Expects JSON body:
    {
        "text": "Text to humanize",
        "mode": "balanced" | "nlp_only" | "ai_only",
        "intensity": "light" | "medium" | "heavy",
        "options": {
            "synonyms": true,
            "contractions": true,
            "vary_length": true,
            "informal": true,
            "casual_starters": true,
            "ai_polish": true
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        mode = data.get('mode', 'balanced')
        intensity = data.get('intensity', 'medium')
        options = data.get('options', {})
        
        result = text
        steps = []
        
        if mode == 'ai_only':
            # Only use AI humanization
            steps.append('AI Humanization')
            result = humanize_with_ai(result, intensity)
            
        elif mode == 'nlp_only':
            # Only use NLP techniques
            steps.append('NLP Processing')
            nlp_options = {
                'synonyms': options.get('synonyms', True),
                'contractions': options.get('contractions', True),
                'vary_length': options.get('vary_length', True),
                'informal': options.get('informal', True),
                'casual_starters': options.get('casual_starters', True),
                'synonym_rate': 0.1 if intensity == 'light' else (0.2 if intensity == 'medium' else 0.3),
                'informal_rate': 0.05 if intensity == 'light' else (0.1 if intensity == 'medium' else 0.15),
            }
            result = humanize_text(result, nlp_options)
            
        else:  # balanced mode
            # Step 1: AI humanization first
            steps.append('AI Humanization')
            result = humanize_with_ai(result, intensity)
            
            # Step 2: Apply NLP techniques for additional variation
            steps.append('NLP Enhancement')
            nlp_options = {
                'synonyms': options.get('synonyms', True),
                'synonym_rate': 0.08,  # Lower rate since AI already made changes
                'contractions': options.get('contractions', True),
                'vary_length': False,  # AI handles this well
                'informal': options.get('informal', False),  # Light touch
                'informal_rate': 0.05,
                'casual_starters': False,  # AI handles this
            }
            result = humanize_text(result, nlp_options)
            
            # Step 3: Optional AI polish
            if options.get('ai_polish', False):
                steps.append('AI Polish')
                result = polish_with_ai(result)
        
        return jsonify({
            'success': True,
            'original': text,
            'humanized': result,
            'mode': mode,
            'intensity': intensity,
            'steps': steps
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'Text Humanizer API'})


if __name__ == '__main__':
    print("Starting Text Humanizer Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
