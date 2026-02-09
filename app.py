"""
FastAPI Text Intelligence Starter - Backend Server

This is a simple FastAPI server that provides a text intelligence API endpoint
powered by Deepgram's Text Intelligence service. It's designed to be easily
modified and extended for your own projects.

Key Features:
- Contract-compliant API endpoint: POST /api/text-intelligence
- Accepts text or URL in JSON body
- Supports multiple intelligence features: summarization, topics, sentiment, intents
- JWT session auth with page nonce (production only)
- Async/await for better performance
- Automatic OpenAPI docs at /docs
"""

import os
import secrets
import time
from typing import Optional

import jwt
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepgram import DeepgramClient
from dotenv import load_dotenv
import toml

# Load .env without overriding existing env vars
load_dotenv(override=False)

CONFIG = {
    "port": int(os.environ.get("PORT", 8081)),
    "host": os.environ.get("HOST", "0.0.0.0"),
}

# ============================================================================
# SESSION AUTH - JWT tokens with page nonce for production security
# ============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
REQUIRE_NONCE = bool(os.environ.get("SESSION_SECRET"))

# In-memory nonce store: nonce -> expiry timestamp
session_nonces = {}
NONCE_TTL = 5 * 60  # 5 minutes
JWT_EXPIRY = 3600  # 1 hour


def generate_nonce():
    """Generates a single-use nonce and stores it with an expiry."""
    nonce = secrets.token_hex(16)
    session_nonces[nonce] = time.time() + NONCE_TTL
    return nonce


def consume_nonce(nonce):
    """Validates and consumes a nonce (single-use). Returns True if valid."""
    expiry = session_nonces.pop(nonce, None)
    if expiry is None:
        return False
    return time.time() < expiry


def cleanup_nonces():
    """Remove expired nonces."""
    now = time.time()
    expired = [k for k, v in session_nonces.items() if now >= v]
    for k in expired:
        del session_nonces[k]


# Read frontend/dist/index.html template for nonce injection
_index_html_template = None
try:
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")) as f:
        _index_html_template = f.read()
except FileNotFoundError:
    pass  # No built frontend (dev mode)


def require_session(authorization: str = Header(None)):
    """FastAPI dependency for JWT session validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required",
                }
            }
        )
    token = authorization[7:]
    try:
        jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Session expired, please refresh the page",
                }
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Invalid session token",
                }
            }
        )


# ============================================================================
# API KEY LOADING
# ============================================================================

def load_api_key():
    """Loads the Deepgram API key from environment variables"""
    api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not api_key:
        print("\n❌ ERROR: Deepgram API key not found!\n")
        print("Please set your API key using one of these methods:\n")
        print("1. Create a .env file (recommended):")
        print("   DEEPGRAM_API_KEY=your_api_key_here\n")
        print("2. Environment variable:")
        print("   export DEEPGRAM_API_KEY=your_api_key_here\n")
        print("Get your API key at: https://console.deepgram.com\n")
        raise ValueError("DEEPGRAM_API_KEY environment variable is required")

    return api_key

api_key = load_api_key()

# ============================================================================
# SETUP
# ============================================================================

deepgram = DeepgramClient(api_key=api_key)

app = FastAPI(
    title="Deepgram Text Intelligence API",
    description="Text analysis powered by Deepgram",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TextInput(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_text_input(body: TextInput):
    """Validates that JSON body has exactly one of text or url"""
    if not body.text and not body.url:
        return None, "Request must contain either 'text' or 'url' field"

    if body.text and body.url:
        return None, "Request must contain either 'text' or 'url', not both"

    if body.url:
        if not body.url.startswith(('http://', 'https://')):
            return None, "Invalid URL format"
        return {"url": body.url}, None
    else:
        if not body.text.strip():
            return None, "Text content cannot be empty"
        return {"text": body.text}, None

def build_deepgram_options(
    language: str = "en",
    summarize: Optional[str] = None,
    topics: Optional[str] = None,
    sentiment: Optional[str] = None,
    intents: Optional[str] = None
):
    """Converts query parameters to SDK keyword arguments"""
    options = {"language": language}

    if summarize == "true":
        options["summarize"] = True
    elif summarize == "v2":
        options["summarize"] = "v2"
    elif summarize == "v1":
        return None, "Summarization v1 is no longer supported. Please use v2 or true."

    if topics == "true":
        options["topics"] = True
    if sentiment == "true":
        options["sentiment"] = True
    if intents == "true":
        options["intents"] = True

    return options, None

# ============================================================================
# SESSION ROUTES - Auth endpoints (unprotected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html with injected session nonce (production only)."""
    if not _index_html_template:
        raise HTTPException(status_code=404, detail="Frontend not built. Run make build first.")
    cleanup_nonces()
    nonce = generate_nonce()
    html = _index_html_template.replace(
        "</head>",
        f'<meta name="session-nonce" content="{nonce}">\n</head>'
    )
    return HTMLResponse(content=html)


@app.get("/api/session")
async def get_session(x_session_nonce: str = Header(None)):
    """Issues a JWT. In production, requires valid nonce via X-Session-Nonce header."""
    if REQUIRE_NONCE:
        if not x_session_nonce or not consume_nonce(x_session_nonce):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "AuthenticationError",
                        "code": "INVALID_NONCE",
                        "message": "Valid session nonce required. Please refresh the page.",
                    }
                }
            )
    token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY},
        SESSION_SECRET,
        algorithm="HS256",
    )
    return JSONResponse(content={"token": token})


# ============================================================================
# API ROUTES
# ============================================================================

@app.post("/api/text-intelligence")
async def analyze(
    body: TextInput,
    language: str = "en",
    summarize: Optional[str] = None,
    topics: Optional[str] = None,
    sentiment: Optional[str] = None,
    intents: Optional[str] = None,
    _auth=Depends(require_session)
):
    """
    POST /api/text-intelligence

    Contract-compliant text intelligence endpoint.
    Accepts JSON body with either text or url field.
    Query parameters: summarize, topics, sentiment, intents, language
    """
    try:
        # Validate text input
        request_dict, error_msg = validate_text_input(body)
        if error_msg:
            error_code = "INVALID_TEXT" if "text" in error_msg.lower() else "INVALID_URL"
            response = JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "validation_error",
                        "code": error_code,
                        "message": error_msg,
                        "details": {}
                    }
                }
            )
            return response

        # Build Deepgram options
        options, error_msg = build_deepgram_options(
            language, summarize, topics, sentiment, intents
        )
        if error_msg:
            response = JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "validation_error",
                        "code": "INVALID_TEXT",
                        "message": error_msg,
                        "details": {}
                    }
                }
            )
            return response

        # Call Deepgram API
        response_data = deepgram.read.v1.text.analyze(
            request=request_dict,
            **options
        )

        # Convert Pydantic model to dict
        if hasattr(response_data, 'to_dict'):
            result = {"results": response_data.results.to_dict() if hasattr(response_data.results, 'to_dict') else {}}
        elif hasattr(response_data, 'model_dump'):
            result_data = response_data.model_dump()
            result = {"results": result_data.get('results', {})}
        else:
            result = {"results": dict(response_data.results) if hasattr(response_data, 'results') else {}}

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        print(f"Text Intelligence Error: {e}")

        error_code = "INVALID_TEXT"
        error_message = str(e)
        status_code = 500

        if "text" in str(e).lower():
            error_code = "INVALID_TEXT"
            status_code = 400
        elif "url" in str(e).lower():
            error_code = "INVALID_URL"
            status_code = 400
        elif "too long" in str(e).lower():
            error_code = "TEXT_TOO_LONG"
            status_code = 400

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": "processing_error",
                    "code": error_code,
                    "message": error_message if status_code == 400 else "Text processing failed",
                    "details": {}
                }
            }
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "text-intelligence"}

@app.get("/api/metadata")
async def get_metadata():
    """
    GET /api/metadata

    Returns metadata about this starter application from deepgram.toml
    """
    try:
        with open('deepgram.toml', 'r') as f:
            config = toml.load(f)

        if 'meta' not in config:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'INTERNAL_SERVER_ERROR',
                    'message': 'Missing [meta] section in deepgram.toml'
                }
            )

        return JSONResponse(content=config['meta'], status_code=200)

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_SERVER_ERROR',
                'message': 'deepgram.toml file not found'
            }
        )

    except Exception as e:
        print(f"Error reading metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_SERVER_ERROR',
                'message': f'Failed to read metadata from deepgram.toml: {str(e)}'
            }
        )

# ============================================================================
# FRONTEND SERVING
# ============================================================================

# ============================================================================
# SERVER START
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    nonce_status = " (nonce required)" if REQUIRE_NONCE else ""
    print("\n" + "=" * 70)
    print(f"🚀 FastAPI Text Intelligence Server running at http://localhost:{CONFIG['port']}")
    print("=" * 70)
    print("\nAvailable routes:")
    print(f"  GET  /api/session{nonce_status}")
    print(f"  POST /api/text-intelligence (auth required)")
    print(f"  GET  /api/metadata")
    print(f"  GET  /health")
    print(f"  GET  /docs (OpenAPI documentation)")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
