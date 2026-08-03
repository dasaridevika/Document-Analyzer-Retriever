import os
import logging
import httpx
import re
import json
from json import JSONDecodeError
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Default Live Cloudflare Worker Endpoint
DEFAULT_WORKER_URL = "https://doc-analyser-worker.devika-worker.workers.dev"

# Base Worker URL (Loaded from environment variables with fallback to live worker endpoint)
WORKER_BASE_URL = os.getenv("LLM_ANALYSIS_URL", "").strip().rstrip("/") or DEFAULT_WORKER_URL

WORKER_ANALYZE_URL = f"{WORKER_BASE_URL}/analyze" if not WORKER_BASE_URL.endswith("/analyze") else WORKER_BASE_URL
WORKER_EMBEDDINGS_URL = f"{WORKER_BASE_URL}/embeddings"

MAX_ANALYSIS_CHARS = 12000
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))

DEFAULT_ANALYSIS = {
    "summary": "",
    "topics": [],
    "keywords": [],
    "sentiment": "neutral",
    "important_points": [],
    "action_items": [],
}

try:
    import repairjson
except Exception:
    repairjson = None


def _safe_analysis(error: str = "", raw_response: str = "") -> dict:
    payload = DEFAULT_ANALYSIS.copy()
    if error:
        payload["error"] = error
    if raw_response:
        payload["raw_response"] = raw_response[:4000]
    return payload
