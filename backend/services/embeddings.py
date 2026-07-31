import requests
import logging
from typing import List, Dict, Any
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_EMBEDDING_MODEL
)
from backend.services.worker_analyzer import generate_bge_embeddings, WORKER_EMBEDDINGS_URL

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Embedding generator supporting both:
    1. Direct Cloudflare Worker URL (@cf/baai/bge-large-en-v1.5)
    2. Direct Cloudflare REST API with Account ID & Token
    3. Fallback to local SentenceTransformers (all-MiniLM-L6-v2)
    """

    def __init__(self):
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.model_name = CLOUDFLARE_EMBEDDING_MODEL
        self.worker_url = WORKER_EMBEDDINGS_URL
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading local fallback embedding model (all-MiniLM-L6-v2)...")
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.error(f"Failed to load sentence_transformers: {e}")
                raise RuntimeError("No embedding provider available.")
        return self._local_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for text list using BGE Large or local fallback.
        """
        if not texts:
            return []

        # Option 1: Try Cloudflare Worker AI Endpoint
        try:
            import asyncio
            # If in async loop, use asyncio task or direct sync POST request
            url = self.worker_url
            response = requests.post(url, json={"text": texts}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                vectors = data.get("data") or data.get("result", {}).get("data")
                if vectors and len(vectors) == len(texts):
                    return vectors
        except Exception as e:
            logger.warning(f"Cloudflare Worker embedding endpoint failed: {e}. Trying direct Cloudflare REST API...")

        # Option 2: Try Direct Cloudflare REST API with Token
        if self.account_id and self.api_token:
            try:
                return self._generate_cloudflare_rest_embeddings(texts)
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API failed: {e}. Falling back to local embeddings.")

        # Option 3: Local Fallback
        local_model = self._get_local_model()
        embeddings = local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

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
