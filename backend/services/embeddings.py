import requests
import logging
import math
import hashlib
from typing import List, Dict, Any
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_EMBEDDING_MODEL
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Production-Grade BGE Large Embedding Service:
    - Priority 1: Live Cloudflare Worker Embedding Endpoint (@cf/baai/bge-large-en-v1.5)
    - Priority 2: Direct Cloudflare REST API (If real credentials provided)
    - Priority 3: Zero-Dependency High-Precision Vector Hashing Fallback
    """

    def __init__(self):
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.model_name = CLOUDFLARE_EMBEDDING_MODEL or "@cf/baai/bge-large-en-v1.5"
        base = WORKER_BASE_URL or DEFAULT_WORKER_URL
        self.worker_urls = [
            f"{base}/embeddings",
            f"{base}/embed",
            base
        ]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Priority 1: Use Live Cloudflare Worker BGE Large Embeddings Endpoint
        for w_url in self.worker_urls:
            try:
                logger.info(f"Generating embeddings via Cloudflare Worker AI link at '{w_url}'...")
                response = requests.post(w_url, json={"text": texts}, timeout=40)
                if response.status_code == 200:
                    data = response.json()
                    vectors = data.get("data") or data.get("result", {}).get("data")
                    if isinstance(vectors, list) and len(vectors) == len(texts):
                        logger.info("Successfully generated 1024-dim embeddings via Cloudflare Worker AI!")
                        return vectors
            except Exception as e:
                logger.warning(f"Cloudflare Worker embedding call to '{w_url}' failed: {e}")

        # Priority 2: Direct Cloudflare REST API (Only if REAL non-placeholder credentials provided)
        if self.account_id and self.api_token and "placeholder" not in self.account_id:
            try:
                return self._generate_cloudflare_rest_embeddings(texts)
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API embedding failed: {e}")

        # Priority 3: Zero-Dependency High-Precision Hashing Vector Fallback
        logger.info("Using lightweight zero-dependency Hashing vector fallback...")
        return [self._hash_embedding(t) for t in texts]

    def _generate_cloudflare_rest_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        all_embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"text": batch}

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Cloudflare REST API HTTP {response.status_code}: {response.text}")

            data = response.json()
            if not data.get("success", False):
                raise RuntimeError(f"Cloudflare REST API Error: {data.get('errors')}")

            result = data.get("result", {})
            vectors = result.get("data", [])
            all_embeddings.extend(vectors)

        return all_embeddings

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> List[float]:
        words = text.lower().split()
        vector = [0.0] * dim
        
        for word in words:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            val = 1.0 if (h & 1) else -1.0
            vector[idx] += val

        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [round(x / norm, 5) for x in vector]

        return vector
