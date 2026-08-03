import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

# FAISS Persistent Index & Metadata Paths
FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
FAISS_META_PATH = VECTOR_DB_DIR / "faiss_metadata.json"

class VectorStoreManager:
    """
    FAISS (Facebook AI Similarity Search) Persistent Vector Store Engine.
    Uses L2-Normalized Cosine Inner Product (IndexFlatIP) for 100% Exact Similarity Search.
    """

    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.documents_store: List[str] = []
        self.ids_store: List[str] = []

        self._init_faiss_engine()

    def _init_faiss_engine(self):
        try:
            import faiss
            self.faiss_module = faiss
        except ImportError:
            logger.warning("FAISS library not installed. Using High-Performance NumPy Cosine Similarity Vector Index.")
            self.faiss_module = None

        if self.faiss_module:
            if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
                try:
                    self.faiss_index = self.faiss_module.read_index(str(FAISS_INDEX_PATH))
                    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        self.metadata_store = meta_data.get("metadatas", [])
                        self.documents_store = meta_data.get("documents", [])
                        self.ids_store = meta_data.get("ids", [])
                    logger.info(f"Loaded existing FAISS Index with {self.faiss_index.ntotal} vectors from '{FAISS_INDEX_PATH}'.")
                    return
                except Exception as e:
                    logger.warning(f"Error loading existing FAISS index: {e}. Initializing new FAISS IndexFlatIP.")

            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
            logger.info(f"Initialized new FAISS IndexFlatIP (dim={self.embedding_dim}).")

    def _normalize_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def _save_persistent_faiss(self):
        try:
            if self.faiss_module and self.faiss_index:
                self.faiss_module.write_index(self.faiss_index, str(FAISS_INDEX_PATH))

            meta_data = {
                "ids": self.ids_store,
                "documents": self.documents_store,
                "metadatas": self.metadata_store
            }
            with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            logger.info("Saved FAISS Index and Metadata to disk successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS persistent index: {e}")

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        if not chunks or not embeddings:
            return

        norm_embeddings = self._normalize_vectors(embeddings)

        if self.faiss_module and self.faiss_index:
            # Check dimensions match
            if norm_embeddings.shape[1] != self.embedding_dim:
                self.embedding_dim = norm_embeddings.shape[1]
                self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)

            self.faiss_index.add(norm_embeddings)
        else:
            if hasattr(self, "numpy_vectors") and self.numpy_vectors is not None:
                self.numpy_vectors = np.vstack([self.numpy_vectors, norm_embeddings])
            else:
                self.numpy_vectors = norm_embeddings

        for c in chunks:
            self.ids_store.append(c["chunk_id"])
            self.documents_store.append(c["text"])
            self.metadata_store.append({
                "filename": c["filename"],
                "page_number": int(c["page_number"]),
                "chunk_index": int(c["chunk_index"]),
                "strategy": c["strategy"],
                "char_count": int(c["char_count"]),
                "word_count": int(c["word_count"])
            })

        self._save_persistent_faiss()
        logger.info(f"Successfully added {len(chunks)} chunks to FAISS Index.")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filename_filter: str = None
    ) -> List[Dict[str, Any]]:
        if not self.ids_store:
            return []

        norm_query = self._normalize_vectors([query_embedding])

        retrieved_chunks = []

        if self.faiss_module and self.faiss_index and self.faiss_index.ntotal > 0:
            search_k = min(self.faiss_index.ntotal, max(top_k * 4, 30))
            scores, indices = self.faiss_index.search(norm_query, search_k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.ids_store):
                    continue
                meta = self.metadata_store[idx]
                if filename_filter and meta.get("filename") != filename_filter:
                    continue

                sim_score = round(float(max(0.0, min(1.0, score))), 4)
                retrieved_chunks.append({
                    "chunk_id": self.ids_store[idx],
                    "text": self.documents_store[idx],
                    "metadata": meta,
                    "similarity_score": sim_score,
                    "distance": 1.0 - sim_score
                })

                if len(retrieved_chunks) >= top_k:
                    break
        else:
            # Fallback NumPy Cosine Similarity Search
            if hasattr(self, "numpy_vectors") and self.numpy_vectors is not None:
                sims = np.dot(self.numpy_vectors, norm_query.T).flatten()
                sorted_indices = np.argsort(sims)[::-1]
                for idx in sorted_indices:
                    meta = self.metadata_store[idx]
                    if filename_filter and meta.get("filename") != filename_filter:
                        continue
                    sim_score = round(float(max(0.0, min(1.0, sims[idx]))), 4)
                    retrieved_chunks.append({
                        "chunk_id": self.ids_store[idx],
                        "text": self.documents_store[idx],
                        "metadata": meta,
                        "similarity_score": sim_score,
                        "distance": 1.0 - sim_score
                    })
                    if len(retrieved_chunks) >= top_k:
                        break

        return retrieved_chunks

    def get_distributed_chunks(
        self,
        filename: str = None,
        count: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Retrieves evenly distributed chunks across the entire document for complete FAISS summarization.
        """
        matching_indices = [
            i for i, m in enumerate(self.metadata_store)
            if not filename or m.get("filename") == filename
        ]

        if not matching_indices:
            return []

        total = len(matching_indices)
        step = max(1, total // count)

        distributed = []
        for i in range(0, total, step):
            idx = matching_indices[i]
            distributed.append({
                "chunk_id": self.ids_store[idx],
                "text": self.documents_store[idx],
                "metadata": self.metadata_store[idx],
                "similarity_score": 0.99
            })
            if len(distributed) >= count:
                break

        return distributed

    def get_page_chunks(
        self,
        filename: str = None,
        pages: List[int] = [1, 2, 3, 4],
        limit: int = 6
    ) -> List[Dict[str, Any]]:
        retrieved = []
        for i, meta in enumerate(self.metadata_store):
            if filename and meta.get("filename") != filename:
                continue
            if meta.get("page_number") in pages:
                retrieved.append({
                    "chunk_id": self.ids_store[i],
                    "text": self.documents_store[i],
                    "metadata": meta,
                    "similarity_score": 0.99
                })
                if len(retrieved) >= limit:
                    break
        return retrieved

    def clear_document(self, filename: str) -> None:
        keep_indices = [
            i for i, m in enumerate(self.metadata_store)
            if m.get("filename") != filename
        ]

        if len(keep_indices) == len(self.metadata_store):
            return

        self.ids_store = [self.ids_store[i] for i in keep_indices]
        self.documents_store = [self.documents_store[i] for i in keep_indices]
        self.metadata_store = [self.metadata_store[i] for i in keep_indices]

        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
            if hasattr(self, "numpy_vectors") and self.numpy_vectors is not None:
                if len(keep_indices) > 0:
                    self.numpy_vectors = self.numpy_vectors[keep_indices]
                    self.faiss_index.add(self.numpy_vectors)
                else:
                    self.numpy_vectors = None

        self._save_persistent_faiss()
        logger.info(f"Cleared chunks for document '{filename}' from FAISS Index.")

    def clear_all(self) -> None:
        self.ids_store = []
        self.documents_store = []
        self.metadata_store = []
        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
        self._save_persistent_faiss()

    def get_stats(self) -> Dict[str, Any]:
        count = len(self.ids_store)
        return {
            "total_chunks": count,
            "vector_engine": "FAISS (Facebook AI Similarity Search)",
            "index_type": "IndexFlatIP (Exact Cosine Similarity)",
            "embedding_dim": self.embedding_dim,
            "storage_directory": str(VECTOR_DB_DIR)
        }
