"""
FastAPI Text Intelligence Starter - Backend Server

This is a simple FastAPI server that provides a text intelligence API endpoint
powered by Deepgram's Text Intelligence service. It's designed to be easily
modified and extended for your own projects.

Key Features:
- Contract-compliant API endpoint: POST /text-intelligence/analyze
- Accepts text or URL in JSON body
- Supports multiple intelligence features: summarization, topics, sentiment, intents
- Async/await for better performance
- Automatic OpenAPI docs at /docs
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
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
    "frontend_port": int(os.environ.get("FRONTEND_PORT", 8080)),
}

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
    allow_origins=[
        f"http://localhost:{CONFIG['frontend_port']}",
        f"http://127.0.0.1:{CONFIG['frontend_port']}",
    ],
    allow_credentials=True,
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
# API ROUTES
# ============================================================================

@app.post("/text-intelligence/analyze")
async def analyze(
    body: TextInput,
    language: str = "en",
    summarize: Optional[str] = None,
    topics: Optional[str] = None,
    sentiment: Optional[str] = None,
    intents: Optional[str] = None,
    x_request_id: Optional[str] = Header(None)
):
    """
    POST /text-intelligence/analyze

    Contract-compliant text intelligence endpoint.
    Accepts JSON body with either text or url field.
    Query parameters: summarize, topics, sentiment, intents, language
    Header: X-Request-Id (optional, echoed back)
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
            if x_request_id:
                response.headers["X-Request-Id"] = x_request_id
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
            if x_request_id:
                response.headers["X-Request-Id"] = x_request_id
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

        response = JSONResponse(status_code=200, content=result)
        if x_request_id:
            response.headers["X-Request-Id"] = x_request_id
        return response

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

        response = JSONResponse(
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
        if x_request_id:
            response.headers["X-Request-Id"] = x_request_id
        return response

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

    print("\n" + "=" * 70)
    print(f"🚀 FastAPI Text Intelligence Server running at http://localhost:{CONFIG['port']}")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
