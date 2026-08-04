import httpx
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_EMBEDDING_MODEL,
    DATA_DIR
)
from backend.services.worker_analyzer import WORKER_BASE_URL, DEFAULT_WORKER_URL

# Optional imports for local ONNX sentence transformer fallback
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import numpy as np
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Production-Grade BGE Large Embedding Service (1024-Dimension):
    - Priority 1: Live Cloudflare Worker Embedding Endpoint (@cf/baai/bge-large-en-v1.5)
    - Priority 2: Direct Cloudflare REST API (Parallelized HTTP via ThreadPoolExecutor)
    - Priority 3: Local ONNX Transformer Fallback (all-MiniLM-L6-v2)
    - Priority 4: 1024-Dimension Feature Hashing Fallback (Guarantees FAISS dimension alignment)
    """

    def __init__(self):
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.model_name = CLOUDFLARE_EMBEDDING_MODEL or "@cf/baai/bge-large-en-v1.5"
        self.expected_dim = 1024
        base = WORKER_BASE_URL or DEFAULT_WORKER_URL
        self.worker_urls = [
            f"{base}/embeddings",
            f"{base}/embed",
            base
        ]
        self.local_onnx_model = None
        if HAS_ONNX:
            try:
                self.local_onnx_model = ONNXEmbeddingModel()
            except Exception as e:
                logger.warning(f"Could not instantiate local ONNX embedding model: {e}")

    def validate_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Validates that every embedding vector matches the expected dimension.
        """
        validated = []
        for vec in vectors:
            if not isinstance(vec, list) or len(vec) != self.expected_dim:
                logger.warning(f"Vector dimension mismatch: expected {self.expected_dim}, got {len(vec) if isinstance(vec, list) else type(vec)}. Re-aligning vector.")
                aligned = self._hash_embedding(" ".join(str(x) for x in vec[:10]), dim=self.expected_dim) if isinstance(vec, list) else self._hash_embedding("", dim=self.expected_dim)
                validated.append(aligned)
            else:
                validated.append(vec)
        return validated

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Priority 1: Cloudflare Worker Endpoint
        for w_url in self.worker_urls:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(w_url, json={"text": texts})
                    if response.status_code == 200:
                        data = response.json()
                        vectors = data.get("data") or data.get("result", {}).get("data")
                        if isinstance(vectors, list) and len(vectors) == len(texts):
                            logger.info(f"Successfully generated {len(vectors)} embeddings via Cloudflare Worker.")
                            return self.validate_vectors(vectors)
                    elif response.status_code in [401, 403]:
                        logger.warning(f"Cloudflare Worker embedding endpoint returned auth error HTTP {response.status_code}.")
                    elif response.status_code == 429:
                        logger.warning("Cloudflare Worker embedding endpoint rate-limited (HTTP 429).")
                    else:
                        logger.warning(f"Cloudflare Worker '{w_url}' returned HTTP {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"Cloudflare Worker embedding call failed on '{w_url}': {e}")

        # Priority 2: Direct Cloudflare REST API (Parallelized HTTP via ThreadPoolExecutor)
        if self.account_id and self.api_token and "placeholder" not in str(self.account_id).lower():
            try:
                vectors = self._generate_cloudflare_rest_embeddings(texts)
                logger.info(f"Successfully generated {len(vectors)} embeddings via Direct Cloudflare REST API.")
                return self.validate_vectors(vectors)
            except Exception as e:
                logger.warning(f"Direct Cloudflare REST API embedding failed: {e}")
        else:
            logger.warning("Direct Cloudflare REST API skipped: Missing or placeholder CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN.")

        # Priority 3: Local ONNX Transformer Fallback (all-MiniLM-L6-v2)
        if HAS_ONNX and self.local_onnx_model:
            try:
                logger.info("Using local ONNX transformer (all-MiniLM-L6-v2) fallback model...")
                local_vecs = self.local_onnx_model.encode(texts)
                return self.validate_vectors(local_vecs)
            except Exception as e:
                logger.warning(f"Local ONNX embedding generation failed: {e}. Falling back to n-gram hashing.")

        # Priority 4: Guaranteed 1024-dim Deterministic Hashing Fallback
        logger.info("Using 1024-dimension Feature Hashing vector fallback...")
        fallback_vecs = [self._hash_embedding(t, dim=self.expected_dim) for t in texts]
        return self.validate_vectors(fallback_vecs)

    def _generate_cloudflare_rest_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        all_embeddings = []
        batch_size = 32
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        def fetch_batch(batch_texts):
            with httpx.Client(timeout=30.0) as client:
                payload = {"text": batch_texts}
                response = client.post(url, headers=headers, json=payload)
                if response.status_code == 401:
                    raise PermissionError("Cloudflare REST API 401 Unauthorized: Invalid API Token.")
                if response.status_code == 404:
                    raise ValueError(f"Cloudflare REST API 404 Not Found: Account ID or Model invalid.")
                if response.status_code == 429:
                    raise RuntimeError("Cloudflare REST API 429 Rate Limit Exceeded.")
                if response.status_code != 200:
                    raise RuntimeError(f"Cloudflare REST API HTTP {response.status_code}: {response.text}")

                data = response.json()
                if not data.get("success", False):
                    raise RuntimeError(f"Cloudflare REST API returned unsuccessful status: {data.get('errors')}")

                result = data.get("result", {})
                return result.get("data", [])

        # Run batches in parallel using ThreadPoolExecutor
        max_workers = min(8, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(fetch_batch, batches))

        for r in results:
            all_embeddings.extend(r)

        return all_embeddings

    @staticmethod
    def _hash_embedding(text: str, dim: int = 1024) -> List[float]:
        """
        Generates a 1024-dimensional normalized n-gram feature hashing vector.
        Matches BGE-Large dimensionality.
        """
        words = text.lower().split()
        vector = [0.0] * dim
        
        for word in words:
            h = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            val = 1.0 if (h & 1) else -1.0
            vector[idx] += val

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


class ONNXEmbeddingModel:
    """
    Lightweight local embedding generator using ONNX runtime and Xenova/all-MiniLM-L6-v2.
    Pads the output to 1024 dimensions to match the BGE-Large expected vector store constraints.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (DATA_DIR / "onnx_model")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / "model.onnx"
        self.tokenizer_path = self.cache_dir / "tokenizer.json"
        self.ort_session = None
        self.tokenizer = None

    def _download_files(self):
        import urllib.request
        model_url = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
        tokenizer_url = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
        
        try:
            if not self.model_path.exists():
                logger.info(f"Downloading ONNX model file to {self.model_path}...")
                urllib.request.urlretrieve(model_url, self.model_path)
        except Exception as e:
            logger.error(f"Failed to download ONNX model file: {e}")
            raise

        try:
            if not self.tokenizer_path.exists():
                logger.info(f"Downloading tokenizer configuration to {self.tokenizer_path}...")
                urllib.request.urlretrieve(tokenizer_url, self.tokenizer_path)
        except Exception as e:
            logger.error(f"Failed to download tokenizer configuration: {e}")
            raise

    def load(self):
        self._download_files()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.ort_session = ort.InferenceSession(str(self.model_path))

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not self.ort_session or not self.tokenizer:
            self.load()

        embeddings = []
        for text in texts:
            encoded = self.tokenizer.encode(text)
            input_ids = encoded.ids
            attention_mask = encoded.attention_mask

            in_ids = np.array([input_ids], dtype=np.int64)
            attn_mask = np.array([attention_mask], dtype=np.int64)
            
            model_inputs = {input.name: input for input in self.ort_session.get_inputs()}
            inputs = {
                "input_ids": in_ids,
                "attention_mask": attn_mask
            }
            if "token_type_ids" in model_inputs:
                inputs["token_type_ids"] = np.zeros((1, len(input_ids)), dtype=np.int64)

            outputs = self.ort_session.run(None, inputs)
            token_embeddings = outputs[0]  # Shape: [1, seq_len, 384]

            # Mean pooling with attention mask
            mask = np.expand_dims(attn_mask, axis=-1)  # Shape: [1, seq_len, 1]
            token_embeddings_masked = token_embeddings * mask
            sum_embeddings = np.sum(token_embeddings_masked, axis=1)
            sum_mask = np.sum(mask, axis=1)
            sum_mask = np.maximum(sum_mask, 1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 Normalize
            norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-9)
            normalized = mean_pooled / norm

            # Pad 384 dimension to 1024 dimension to match BGE-Large expected vector store constraints
            vector_384 = normalized[0].tolist()
            vector_1024 = vector_384 + [0.0] * (1024 - 384)
            embeddings.append(vector_1024)

        return embeddings
