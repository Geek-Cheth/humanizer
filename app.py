"""
Text Humanizer API Server
Flask backend with Clerk authentication and OpenRouter AI.

Pipeline per pass (all via OpenRouter — no other API keys needed):
  A. Triple-hop translation  EN->ES->FR->EN  [Llama 4 Maverick]
  B. NLP transforms          cliches, spellings, clause flips, homoglyphs
  C. Perplexity boost        vocabulary disruption  [DeepSeek V3]
  D. Structural burstiness   sentence-length variance  [DeepSeek V3]
  E. Humanity injection      parentheticals, hedging  [DeepSeek V3]
  F. Naturalness smoother    fix stiff/archaic words  [DeepSeek V3]

Two different model families (Llama + DeepSeek) through one OpenRouter key
preserves cross-model fingerprint mixing without multiple API providers.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import os
import jwt
import requests
from dotenv import load_dotenv

from humanizer import humanize_text, humanize_text_academic, remove_ai_cliches
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
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'error': 'Authentication required', 'code': 'AUTH_REQUIRED'}), 401
        token = auth_header.replace('Bearer ', '')
        payload = verify_clerk_token(token)
        if not payload:
            return jsonify({'success': False, 'error': 'Invalid or expired token', 'code': 'INVALID_TOKEN'}), 401
        request.user_id = payload.get('sub')
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


@app.route('/api/humanize', methods=['POST'])
@require_auth
def humanize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        style  = data.get('style', 'academic')
        passes = max(1, min(5, int(data.get('passes', 2))))

        academic_opts = {'vary_length': True, 'contractions': False, 'informal': False, 'casual_starters': False}
        casual_opts   = {'vary_length': True, 'contractions': True,  'informal': True,  'informal_rate': 0.15, 'casual_starters': True}

        nlp_opts = academic_opts if style == 'academic' else casual_opts
        nlp_fn   = humanize_text_academic if style == 'academic' else humanize_text

        result = text
        steps  = []

        result = remove_ai_cliches(result)
        steps.append('AI Cliche Removal (NLP)')

        for i in range(passes):
            n = i + 1

            result = triple_translation(result)
            steps.append(f'Pass {n}/{passes} - Translation EN->ES->FR->EN (Llama 4 Maverick)')

            result = nlp_fn(result, nlp_opts)
            steps.append(f'Pass {n}/{passes} - NLP: Spellings, Clause Flips, Homoglyphs')

            result = perplexity_boost(result)
            steps.append(f'Pass {n}/{passes} - Perplexity Boost (DeepSeek V3)')

            result = structural_restructure(result)
            steps.append(f'Pass {n}/{passes} - Structural Burstiness (DeepSeek V3)')

            result = humanity_injection(result, style)
            steps.append(f'Pass {n}/{passes} - Humanity Injection (DeepSeek V3)')

            result = naturalness_smoother(result)
            steps.append(f'Pass {n}/{passes} - Naturalness Smoother (DeepSeek V3)')

        result = format_only(result)
        steps.append('Final - Spacing & Punctuation Fix')

        return jsonify({
            'success':   True,
            'original':  text,
            'humanized': result,
            'style':     style,
            'passes':    passes,
            'steps':     steps,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'Text Humanizer API'})


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
    print("Starting Text Humanizer Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
