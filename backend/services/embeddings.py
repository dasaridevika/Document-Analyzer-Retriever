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
    Production-Grade BGE Large Embedding Service (1024-Dimension):
    - Priority 1: Live Cloudflare Worker Embedding Endpoint (@cf/baai/bge-large-en-v1.5)
    - Priority 2: Direct Cloudflare REST API (If credentials provided)
    - Priority 3: 1024-Dimension Feature Hashing Fallback (Guarantees FAISS dimension alignment)
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

        # Priority 1: Cloudflare Worker Endpoint
        for w_url in self.worker_urls:
            try:
                response = requests.post(w_url, json={"text": texts}, timeout=40)
                if response.status_code == 200:
                    data = response.json()
                    vectors = data.get("data") or data.get("result", {}).get("data")
                    if isinstance(vectors, list) and len(vectors) == len(texts):
                        return vectors
            except Exception as e:
                logger.warning(f"Cloudflare Worker embedding call to '{w_url}' failed: {e}")

        # Priority 2: Direct Cloudflare REST API
        if self.account_id and self.api_token and "placeholder" not in self.account_id:
            try:
                return self._generate_cloudflare_rest_embeddings(texts)
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API embedding failed: {e}")

        # Priority 3: Guaranteed 1024-dim Deterministic Hashing Fallback
        logger.info("Using 1024-dimension Feature Hashing vector fallback...")
        return [self._hash_embedding(t, dim=1024) for t in texts]

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
    def _hash_embedding(text: str, dim: int = 1024) -> List[float]:
        """
        Generates a 1024-dimensional normalized n-gram feature hashing vector.
        Matches BGE-Large dimensionality.
        """
        words = text.lower().split()
        vector = [0.0] * dim
        
        # Word-level features
        for word in words:
            h = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            val = 1.0 if (h & 1) else -1.0
            vector[idx] += val

        # Character bi-gram features for subword semantics
        clean_str = text.lower()
        for i in range(len(clean_str) - 2):
            bigram = clean_str[i:i+3]
            h = int(hashlib.md5(bigram.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            val = 0.5 if (h & 1) else -0.5
            vector[idx] += val

        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [round(x / norm, 5) for x in vector]

        return vector
