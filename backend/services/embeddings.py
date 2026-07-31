import requests
import logging
from typing import List, Dict, Any
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Embedding generator connecting to Cloudflare Workers AI REST API
    (e.g., @cf/baai/bge-large-en-v1.5 or @cf/google/embeddinggemma-300m)
    with automatic fallback to local sentence-transformers.
    """

    def __init__(self):
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.model_name = CLOUDFLARE_EMBEDDING_MODEL
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading local fallback embedding model (all-MiniLM-L6-v2)...")
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.error(f"Failed to load sentence_transformers: {e}")
                raise RuntimeError("Neither Cloudflare Workers AI nor local sentence-transformers are available.")
        return self._local_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text strings.
        Batch size: up to 32 items per call to preserve free tier token limits.
        """
        if not texts:
            return []

        # Check if Cloudflare credentials exist
        if self.account_id and self.api_token:
            try:
                return self._generate_cloudflare_embeddings(texts)
            except Exception as e:
                logger.warning(f"Cloudflare Workers AI Embedding API failed: {e}. Falling back to local embeddings.")

        # Fallback to local model
        local_model = self._get_local_model()
        embeddings = local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def _generate_cloudflare_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calls Cloudflare Workers AI REST API endpoint for vector embeddings.
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        all_embeddings = []
        batch_size = 32  # Batch requests to fit Cloudflare Workers AI payload boundaries

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"text": batch}

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Cloudflare Workers AI API returned HTTP {response.status_code}: {response.text}"
                )

            data = response.json()
            if not data.get("success", False):
                errors = data.get("errors", [])
                raise RuntimeError(f"Cloudflare Workers AI API Error: {errors}")

            result = data.get("result", {})
            data_vectors = result.get("data", [])

            if data_vectors:
                all_embeddings.extend(data_vectors)
            elif "shape" in result and "data" in result:
                # Handle 2D list return format
                all_embeddings.extend(result["data"])
            else:
                raise ValueError(f"Unexpected Cloudflare embedding response format: {result}")

        return all_embeddings
