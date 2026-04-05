"""
Text Humanizer API Server
Flask backend with Clerk authentication and Cerebras AI for text humanization.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import os
import jwt
import requests
from dotenv import load_dotenv

from humanizer import humanize_text, humanize_text_academic, remove_ai_cliches
from cerebras_client import polish_iteration, format_only

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app, origins=['*'], supports_credentials=True)

# Clerk configuration
CLERK_PUBLISHABLE_KEY = os.getenv('NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY', '')
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY', '')

# Cache for Clerk JWKS
_jwks_cache = None


def get_clerk_jwks():
    """Fetch Clerk's JWKS (JSON Web Key Set) for JWT verification."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    
    try:
        # Extract the Clerk frontend API from the publishable key
        # Format: pk_test_<base64 encoded frontend api>
        import base64
        key_data = CLERK_PUBLISHABLE_KEY.replace('pk_test_', '').replace('pk_live_', '')
        # Add padding if needed
        padding = 4 - len(key_data) % 4
        if padding != 4:
            key_data += '=' * padding
        frontend_api = base64.b64decode(key_data).decode('utf-8')
        if frontend_api.endswith('$'):
            frontend_api = frontend_api[:-1]
        
        jwks_url = f"https://{frontend_api}/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        return _jwks_cache
    except Exception as e:
        print(f"Error fetching JWKS: {e}")
        return None


def get_public_key(token):
    """Get the public key from JWKS matching the token's kid."""
    try:
        jwks = get_clerk_jwks()
        if not jwks:
            return None
        
        # Get the key ID from the token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        
        # Find the matching key
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
        
        return None
    except Exception as e:
        print(f"Error getting public key: {e}")
        return None


def verify_clerk_token(token):
    """Verify a Clerk session token and return the payload if valid."""
    try:
        public_key = get_public_key(token)
        if not public_key:
            return None
        
        # Verify the token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            options={'verify_aud': False}  # Clerk doesn't use audience claim
        )
        return payload
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None


def require_auth(f):
    """Decorator to require authentication for an endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'code': 'AUTH_REQUIRED'
            }), 401
        
        token = auth_header.replace('Bearer ', '')
        
        # Verify the token
        payload = verify_clerk_token(token)
        if not payload:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token',
                'code': 'INVALID_TOKEN'
            }), 401
        
        # Add user info to request context
        request.user_id = payload.get('sub')
        request.session_id = payload.get('sid')
        
        return f(*args, **kwargs)
    
    return decorated_function


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS)."""
    return send_from_directory('.', filename)


@app.route('/api/humanize', methods=['POST'])
@require_auth
def humanize():
    """
    Multi-pass humanization endpoint.

    Expects JSON body:
    {
        "text": "Text to humanize",
        "style": "academic" | "casual",
        "passes": 1-5
    }

    Pipeline (per pass):
        1. Translation Shuffle (EN -> DE -> EN)
        2. NLP transform (synonym swap, cliché removal, spelling mix, clause flip)
        3. AI repair  — ONLY fixes broken/nonsensical NLP output, keeps rest verbatim
    Final step:
        AI format-only — corrects spacing/punctuation, zero content changes
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        style = data.get('style', 'academic')       # 'academic' or 'casual'
        passes = max(1, min(5, int(data.get('passes', 2))))  # clamp 1–5

        # ── NLP options per pass ─────────────────────────────────────────────────
        # Always aggressive to aim for 0% detection
        base_synonym_rate = 0.20

        # Academic NLP options (no contractions / informal / casual starters)
        academic_nlp_opts = {
            'synonyms': True,
            'synonym_rate': base_synonym_rate,
            'vary_length': True,
            'contractions': False,
            'informal': False,
            'casual_starters': False,
        }

        # Casual NLP options
        casual_nlp_opts = {
            'synonyms': True,
            'synonym_rate': base_synonym_rate,
            'vary_length': True,
            'contractions': True,
            'informal': True,
            'informal_rate': 0.15,
            'casual_starters': True,
        }

        nlp_opts = academic_nlp_opts if style == 'academic' else casual_nlp_opts
        nlp_fn = humanize_text_academic if style == 'academic' else humanize_text

        result = text
        steps = []

        # ── Strip AI clichés once before any passes ───────────────────────────────
        result = remove_ai_cliches(result)
        steps.append('AI Cliché Removal')

        # ── Multi-pass loop ──────────────────────────────────────────────────────
        for i in range(passes):
            pass_num = i + 1

            # Step A: Translation Shuffle (Lexical Destruction)
            from cerebras_client import translation_shuffle
            result = translation_shuffle(result)
            steps.append(f'Pass {pass_num}/{passes} — Translation Shuffle (EN→DE→EN)')

            # Step B: NLP transformation
            result = nlp_fn(result, nlp_opts)
            steps.append(f'Pass {pass_num}/{passes} — NLP Transform (Clauses & Syntax)')

            # Step C: AI repair — only fix what NLP broke, everything else stays
            result = polish_iteration(result, style)
            steps.append(f'Pass {pass_num}/{passes} — AI Repair')

        # ── Final pass: formatting only — ZERO content changes ────────────────────
        result = format_only(result)
        steps.append('Final — Spacing & Formatting Fix')

        return jsonify({
            'success': True,
            'original': text,
            'humanized': result,
            'style': style,
            'passes': passes,
            'steps': steps
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint (no auth required)."""
    return jsonify({'status': 'ok', 'service': 'Text Humanizer API'})


@app.route('/api/auth/check', methods=['GET'])
def auth_check():
    """Check if the current request is authenticated."""
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return jsonify({'authenticated': False})
    
    token = auth_header.replace('Bearer ', '')
    payload = verify_clerk_token(token)
    
    if payload:
        return jsonify({
            'authenticated': True,
            'userId': payload.get('sub')
        })
    
    return jsonify({'authenticated': False})


if __name__ == '__main__':
    print("Starting Text Humanizer Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
