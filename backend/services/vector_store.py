import os
import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
FAISS_META_PATH = VECTOR_DB_DIR / "faiss_metadata.json"

class BM25Scorer:
    """
    Lightweight, fast BM25 Lexical Scorer for hybrid search re-ranking.
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc.lower().split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(corpus))
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.N = len(corpus)
        self._initialize()

    def _initialize(self):
        df: Dict[str, int] = {}
        for doc in self.corpus:
            frequencies: Dict[str, int] = {}
            for word in doc.lower().split():
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            # BM25 IDF with smoothing
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        q_words = query.lower().split()
        for q in q_words:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / max(1.0, self.avgdl)))
                scores[i] += idf_val * (num / denom)
        return scores


class VectorStoreManager:
    """
    Production-Grade Hybrid Vector Store Engine.
    Combines BGE Dense Vector Cosine Similarity with BM25 Lexical Search via Reciprocal Rank Fusion (RRF).
    Guarantees exact document candidate filtering and semantic intent matching.
    """

    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.documents_store: List[str] = []
        self.ids_store: List[str] = []
        self.vector_matrix: Optional[np.ndarray] = None

        self._init_faiss_engine()

    def _init_faiss_engine(self):
        try:
            import faiss
            self.faiss_module = faiss
        except ImportError:
            logger.warning("FAISS module not installed. Using High-Performance NumPy Cosine Similarity Engine.")
            self.faiss_module = None

        if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
            try:
                if self.faiss_module:
                    self.faiss_index = self.faiss_module.read_index(str(FAISS_INDEX_PATH))
                with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self.metadata_store = meta_data.get("metadatas", [])
                    self.documents_store = meta_data.get("documents", [])
                    self.ids_store = meta_data.get("ids", [])
                    stored_vecs = meta_data.get("vectors")
                    if stored_vecs:
                        self.vector_matrix = np.array(stored_vecs, dtype=np.float32)
                logger.info(f"Loaded existing vector index with {len(self.ids_store)} chunks.")
                return
            except Exception as e:
                logger.warning(f"Error loading vector index: {e}. Initializing clean index.")

        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)

    def _normalize_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
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
                "metadatas": self.metadata_store,
                "vectors": self.vector_matrix.tolist() if self.vector_matrix is not None else []
            }
            with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            logger.info("Saved persistent vector store to disk successfully.")
        except Exception as e:
            logger.error(f"Failed to save persistent vector store: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks or not embeddings:
            return

        norm_embeddings = self._normalize_vectors(embeddings)
        dim = norm_embeddings.shape[1]

        if dim != self.embedding_dim:
            self.embedding_dim = dim
            if self.faiss_module:
                self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)

        if self.faiss_module and self.faiss_index:
            self.faiss_index.add(norm_embeddings)

        if self.vector_matrix is not None:
            self.vector_matrix = np.vstack([self.vector_matrix, norm_embeddings])
        else:
            self.vector_matrix = norm_embeddings

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

    def similarity_search(
        self,
        query_embedding: List[float],
        raw_query: str = "",
        top_k: int = 8,
        filename_filter: str = None,
        min_score: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Semantic Search: Combines BGE Dense Cosine Similarity with BM25 Lexical Keyword Search
        using Reciprocal Rank Fusion (RRF). Guarantees exact target candidate evaluation.
        """
        if not self.ids_store or self.vector_matrix is None:
            return []

        norm_query = self._normalize_vectors([query_embedding])

        if norm_query.shape[1] != self.vector_matrix.shape[1]:
            logger.warning(f"Embedding dimension mismatch: Query ({norm_query.shape[1]}) vs Store ({self.vector_matrix.shape[1]})")
            return []

        candidate_indices = [
            i for i, meta in enumerate(self.metadata_store)
            if not filename_filter or meta.get("filename") == filename_filter
        ]

        if not candidate_indices:
            candidate_indices = list(range(len(self.ids_store)))

        sub_matrix = self.vector_matrix[candidate_indices]
        sub_docs = [self.documents_store[i] for i in candidate_indices]

        # 1. Dense Semantic Cosine Similarity
        dense_sims = np.dot(sub_matrix, norm_query.T).flatten()
        dense_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(dense_sims)[::-1])}

        # 2. Lexical BM25 Search
        bm25_rank_map = {}
        if raw_query.strip() and sub_docs:
            bm25 = BM25Scorer(sub_docs)
            bm25_scores = bm25.get_scores(raw_query)
            bm25_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(bm25_scores)[::-1])}

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = []
        k_rrf = 60
        for loc_idx in range(len(candidate_indices)):
            d_rank = dense_rank_map.get(loc_idx, 999)
            b_rank = bm25_rank_map.get(loc_idx, 999)
            rrf = (1.0 / (k_rrf + d_rank)) + (0.5 / (k_rrf + b_rank))
            rrf_scores.append((rrf, loc_idx, dense_sims[loc_idx]))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)

        retrieved_chunks = []
        for rrf_val, loc_idx, d_score in rrf_scores:
            if d_score < min_score and len(retrieved_chunks) >= 3:
                continue

            orig_idx = candidate_indices[loc_idx]
            retrieved_chunks.append({
                "chunk_id": self.ids_store[orig_idx],
                "text": self.documents_store[orig_idx],
                "metadata": self.metadata_store[orig_idx],
                "similarity_score": round(float(d_score), 4),
                "rrf_score": round(float(rrf_val), 4),
                "distance": round(float(1.0 - d_score), 4)
            })

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def get_distributed_chunks(self, filename: str = None, count: int = 8) -> List[Dict[str, Any]]:
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
                "similarity_score": 0.50
            })
            if len(distributed) >= count:
                break
        return distributed

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

        if self.vector_matrix is not None and len(keep_indices) > 0:
            self.vector_matrix = self.vector_matrix[keep_indices]
        else:
            self.vector_matrix = None

        if self.faiss_module and self.vector_matrix is not None:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(self.vector_matrix)

        self._save_persistent_faiss()

    def clear_all(self) -> None:
        self.ids_store = []
        self.documents_store = []
        self.metadata_store = []
        self.vector_matrix = None
        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
        self._save_persistent_faiss()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.ids_store),
            "vector_engine": "FAISS & NumPy Hybrid Search Engine (RRF BM25 + BGE Dense)",
            "embedding_dim": self.embedding_dim,
            "storage_directory": str(VECTOR_DB_DIR)
        }
