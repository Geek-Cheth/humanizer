"""
Text Humanizer API Server - Modern EventStream Architecture
Flask backend with Clerk authentication, guest mode, and OpenRouter AI pipeline.

Endpoints:
  - POST /api/analyze         Fast AI detection analysis & signal scanner
  - POST /api/humanize        REST humanization pipeline with metric scores
  - POST /api/humanize/stream Server-Sent Events (SSE) streaming humanization
  - GET  /api/health          Health status
"""

import json
import time
from functools import wraps
import os
import jwt
import requests
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

from humanizer import (
    humanize_text,
    humanize_text_academic,
    remove_ai_cliches,
    calculate_ai_score,
    calculate_readability,
    scan_ai_signals,
)
from openrouter_client import (
    triple_translation,
    perplexity_boost,
    structural_restructure,
    humanity_injection,
    naturalness_smoother,
    format_only,
)

load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app, origins=['*'], supports_credentials=True)

CLERK_PUBLISHABLE_KEY = os.getenv('NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY', '')
_jwks_cache = None


def get_clerk_jwks():
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    try:
        import base64
        key_data = CLERK_PUBLISHABLE_KEY.replace('pk_test_', '').replace('pk_live_', '')
        padding = 4 - len(key_data) % 4
        if padding != 4:
            key_data += '=' * padding
        frontend_api = base64.b64decode(key_data).decode('utf-8').rstrip('$')
        jwks_url = f"https://{frontend_api}/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        return _jwks_cache
    except Exception as e:
        print(f"Error fetching JWKS: {e}")
        return None


def get_public_key(token):
    try:
        jwks = get_clerk_jwks()
        if not jwks:
            return None
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
        return None
    except Exception as e:
        print(f"Error getting public key: {e}")
        return None


def verify_clerk_token(token):
    try:
        public_key = get_public_key(token)
        if not public_key:
            return None
        payload = jwt.decode(
            token, public_key,
            algorithms=['RS256'],
            options={'verify_aud': False}
        )
        return payload
    except Exception:
        return None


def require_auth_or_guest(f):
    """Allow full access to authenticated users, or guest mode up to 300 words."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        is_authenticated = False
        user_id = None

        if auth_header.startswith('Bearer '):
            token = auth_header.replace('Bearer ', '')
            payload = verify_clerk_token(token)
            if payload:
                is_authenticated = True
                user_id = payload.get('sub')

        data = request.get_json(silent=True) or {}
        text = data.get('text', '')
        word_count = len(text.split())

        # If not authenticated, check guest word limit (300 words max)
        if not is_authenticated and word_count > 300:
            return jsonify({
                'success': False,
                'error': 'Guest limit exceeded (300 words max). Please sign in to humanize longer documents.',
                'code': 'GUEST_LIMIT_EXCEEDED'
            }), 401

        request.user_id = user_id
        request.is_authenticated = is_authenticated
        return f(*args, **kwargs)

    return decorated


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Fast on-demand AI detection analysis & signal scanning endpoint."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        ai_metrics = calculate_ai_score(text)
        readability = calculate_readability(text)
        signals = scan_ai_signals(text)

        return jsonify({
            'success': True,
            'word_count': readability['word_count'],
            'ai_score': ai_metrics,
            'readability': readability,
            'signals': signals
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/humanize', methods=['POST'])
@require_auth_or_guest
def humanize():
    """Standard REST endpoint for humanization."""
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        style = data.get('style', 'academic')
        passes = max(1, min(5, int(data.get('passes', 2))))

        pre_scores = calculate_ai_score(text)
        pre_readability = calculate_readability(text)

        nlp_opts = {'vary_length': True, 'contractions': style in ['casual', 'executive']}
        nlp_fn = humanize_text_academic if style in ['academic', 'executive'] else humanize_text

        result = text
        steps = []

        result = remove_ai_cliches(result)
        steps.append('AI Signature Removal (NLP)')

        for i in range(passes):
            n = i + 1
            result = triple_translation(result)
            steps.append(f'Pass {n}/{passes} - Triple Translation EN->ES->FR->EN')

            result = nlp_fn(result, nlp_opts)
            steps.append(f'Pass {n}/{passes} - NLP Clause Restructuring & Spelling Mix')

            result = perplexity_boost(result, style=style)
            steps.append(f'Pass {n}/{passes} - Perplexity Boost ({style.replace("_", " ").title()} Mode)')

            result = structural_restructure(result)
            steps.append(f'Pass {n}/{passes} - Sentence Burstiness Disruption')

            result = humanity_injection(result, style=style)
            steps.append(f'Pass {n}/{passes} - Humanity Quirks & Parentheticals Injection')

            result = naturalness_smoother(result)
            steps.append(f'Pass {n}/{passes} - Naturalness Smoother')

        result = format_only(result)
        steps.append('Final - Spacing & Punctuation Polish')

        post_scores = calculate_ai_score(result)
        post_readability = calculate_readability(result)

        return jsonify({
            'success': True,
            'original': text,
            'humanized': result,
            'style': style,
            'passes': passes,
            'steps': steps,
            'pre_scores': pre_scores,
            'post_scores': post_scores,
            'pre_readability': pre_readability,
            'post_readability': post_readability,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/humanize/stream', methods=['POST'])
@require_auth_or_guest
def humanize_stream():
    """Server-Sent Events (SSE) streaming humanization endpoint."""
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    style = data.get('style', 'academic')
    passes = max(1, min(5, int(data.get('passes', 2))))

    def generate_events():
        try:
            pre_scores = calculate_ai_score(text)
            pre_readability = calculate_readability(text)

            yield f"data: {json.dumps({'type': 'init', 'pre_scores': pre_scores, 'pre_readability': pre_readability})}\n\n"

            nlp_opts = {'vary_length': True, 'contractions': style in ['casual', 'executive']}
            nlp_fn = humanize_text_academic if style in ['academic', 'executive'] else humanize_text

            result = text
            total_steps = 1 + (passes * 6) + 1
            current_step = 0

            # Step 0: Cliche Removal
            current_step += 1
            msg = 'AI Signature & Buzzword Removal'
            progress = int((current_step / total_steps) * 100)
            yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
            result = remove_ai_cliches(result)
            yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

            for i in range(passes):
                n = i + 1

                # 1. Translation
                current_step += 1
                msg = f'Pass {n}/{passes} - Cross-Model Translation (EN->ES->FR->EN)'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = triple_translation(result)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

                # 2. NLP
                current_step += 1
                msg = f'Pass {n}/{passes} - Clause Flips & Spelling Variance'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = nlp_fn(result, nlp_opts)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

                # 3. Perplexity
                current_step += 1
                msg = f'Pass {n}/{passes} - Perplexity Boost ({style.replace("_", " ").title()} Mode)'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = perplexity_boost(result, style=style)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

                # 4. Burstiness
                current_step += 1
                msg = f'Pass {n}/{passes} - Sentence Burstiness Disruption'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = structural_restructure(result)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

                # 5. Humanity
                current_step += 1
                msg = f'Pass {n}/{passes} - Authentic Humanity Quirks Injection'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = humanity_injection(result, style=style)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

                # 6. Naturalness
                current_step += 1
                msg = f'Pass {n}/{passes} - Naturalness Smoother'
                progress = int((current_step / total_steps) * 100)
                yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
                result = naturalness_smoother(result)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

            # Final Format
            current_step += 1
            msg = 'Final Formatting & Punctuation Polish'
            progress = 100
            yield f"data: {json.dumps({'type': 'step_start', 'step': msg, 'progress': progress})}\n\n"
            result = format_only(result)
            yield f"data: {json.dumps({'type': 'step_complete', 'step': msg, 'current_text': result})}\n\n"

            post_scores = calculate_ai_score(result)
            post_readability = calculate_readability(result)

            yield f"data: {json.dumps({'type': 'complete', 'humanized': result, 'post_scores': post_scores, 'post_readability': post_readability})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(stream_with_context(generate_events()), mimetype='text/event-stream')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Text Humanizer Stream Engine'})


@app.route('/api/auth/check', methods=['GET'])
def auth_check():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'authenticated': False})
    token = auth_header.replace('Bearer ', '')
    payload = verify_clerk_token(token)
    if payload:
        return jsonify({'authenticated': True, 'userId': payload.get('sub')})
    return jsonify({'authenticated': False})


if __name__ == '__main__':
    print("Starting Text Humanizer Modern Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
