import os
import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import VECTOR_DB_DIR, MAX_CHUNKS_PER_PAGE, DENSE_TOP_K, KEYWORD_TOP_K

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
    Production-Grade Vector Store Engine with Strict Metadata Filtering & Page Frequency Cap.
    """

    def __init__(self, embedding_dim: int = 1024, embedding_model: str = "@cf/baai/bge-large-en-v1.5"):
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
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
                    stored_model = meta_data.get("embedding_model", self.embedding_model)
                    stored_dim = meta_data.get("embedding_dim", self.embedding_dim)

                    if stored_model != self.embedding_model or stored_dim != self.embedding_dim:
                        logger.warning(f"Embedding model mismatch detected (Stored: {stored_model}/{stored_dim} vs Current: {self.embedding_model}/{self.embedding_dim}). Initializing fresh collection.")
                        self._clear_internal()
                        return

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
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "ids": self.ids_store,
                "documents": self.documents_store,
                "metadatas": self.metadata_store,
                "vectors": self.vector_matrix.tolist() if self.vector_matrix is not None else []
            }
            with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
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
                "chunk_id": c["chunk_id"],
                "chunk_index": int(c["chunk_index"]),
                "document_id": c.get("document_id", ""),
                "session_id": c.get("session_id", ""),
                "user_id": c.get("user_id", "anonymous_user"),
                "filename": c.get("filename", c.get("document_name", "")),
                "document_name": c.get("document_name", c.get("filename", "")),
                "document_version": c.get("document_version", "1.0"),
                "page_number": int(c.get("page_number", 1)),
                "section_title": c.get("section_title", "General Section"),
                "strategy": c.get("strategy", "recursive"),
                "extraction_method": c.get("extraction_method", "PyMuPDF Reading Order"),
                "embedding_model": c.get("embedding_model", self.embedding_model),
                "embedding_dimension": int(c.get("embedding_dimension", self.embedding_dim)),
                "created_at": c.get("created_at", "")
            })

        self._save_persistent_faiss()

    def get_first_pages_chunks(self, filename: str = None, document_id: str = None, max_pages: int = 5) -> List[Dict[str, Any]]:
        first_chunks = []
        for i, meta in enumerate(self.metadata_store):
            if (not document_id or meta.get("document_id") == document_id) and \
               (not filename or meta.get("filename") == filename or meta.get("document_name") == filename):
                if meta.get("page_number", 1) <= max_pages:
                    first_chunks.append({
                        "chunk_id": self.ids_store[i],
                        "text": self.documents_store[i],
                        "metadata": meta,
                        "similarity_score": 0.95
                    })
        return first_chunks

    def similarity_search(
        self,
        query_embedding: List[float],
        raw_query: str = "",
        top_k: int = 12,
        filename_filter: str = None,
        document_id_filter: str = None,
        session_id_filter: str = None,
        user_id_filter: str = None,
        min_score: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Step 4 Requirement: Every retrieval request MUST filter by active document_id (and session_id/user_id if provided).
        Step 6 Requirement: Apply MAX_CHUNKS_PER_PAGE limit.
        """
        if not self.ids_store or self.vector_matrix is None:
            return []

        norm_query = self._normalize_vectors([query_embedding])

        if norm_query.shape[1] != self.vector_matrix.shape[1]:
            logger.warning(f"Embedding dimension mismatch: Query ({norm_query.shape[1]}) vs Store ({self.vector_matrix.shape[1]})")
            return []

        candidate_indices = []
        for i, meta in enumerate(self.metadata_store):
            if document_id_filter and meta.get("document_id") and meta.get("document_id") != document_id_filter:
                continue
            if filename_filter and meta.get("filename") != filename_filter and meta.get("document_name") != filename_filter:
                continue
            if session_id_filter and meta.get("session_id") and meta.get("session_id") != session_id_filter:
                continue
            if user_id_filter and meta.get("user_id") and meta.get("user_id") != user_id_filter:
                continue
            candidate_indices.append(i)

        if not candidate_indices:
            return []

        sub_matrix = self.vector_matrix[candidate_indices]
        sub_docs = [self.documents_store[i] for i in candidate_indices]

        # 1. Dense Cosine Similarity
        dense_sims = np.dot(sub_matrix, norm_query.T).flatten()
        dense_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(dense_sims)[::-1])}

        # 2. Lexical BM25 Search
        bm25_rank_map = {}
        if raw_query.strip() and sub_docs:
            bm25 = BM25Scorer(sub_docs)
            bm25_scores = bm25.get_scores(raw_query)
            bm25_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(bm25_scores)[::-1])}

        # 3. Reciprocal Rank Fusion
        rrf_scores = []
        k_rrf = 60
        for loc_idx in range(len(candidate_indices)):
            d_rank = dense_rank_map.get(loc_idx, 999)
            b_rank = bm25_rank_map.get(loc_idx, 999)
            rrf = (1.0 / (k_rrf + d_rank)) + (0.5 / (k_rrf + b_rank))
            rrf_scores.append((rrf, loc_idx, dense_sims[loc_idx]))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)

        retrieved_chunks = []
        page_counts: Dict[int, int] = {}

        for rrf_val, loc_idx, d_score in rrf_scores:
            if d_score < min_score and len(retrieved_chunks) >= 3:
                continue

            orig_idx = candidate_indices[loc_idx]
            meta = self.metadata_store[orig_idx]
            p_num = meta.get("page_number", 1)

            # Cap chunks per page
            if page_counts.get(p_num, 0) >= MAX_CHUNKS_PER_PAGE:
                continue
            page_counts[p_num] = page_counts.get(p_num, 0) + 1

            retrieved_chunks.append({
                "chunk_id": self.ids_store[orig_idx],
                "text": self.documents_store[orig_idx],
                "metadata": meta,
                "similarity_score": round(float(d_score), 4),
                "rrf_score": round(float(rrf_val), 4),
                "distance": round(float(1.0 - d_score), 4)
            })

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def get_neighbor_chunks(self, document_id: str, page_number: int, chunk_index: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i, meta in enumerate(self.metadata_store):
            if meta.get("document_id") == document_id or meta.get("filename") == document_id:
                if meta.get("page_number") == page_number and abs(meta.get("chunk_index", 0) - chunk_index) == 1:
                    neighbors.append({
                        "chunk_id": self.ids_store[i],
                        "text": self.documents_store[i],
                        "metadata": meta
                    })
        return neighbors

    def get_distributed_chunks(self, filename: str = None, document_id: str = None, count: int = 8) -> List[Dict[str, Any]]:
        matching_indices = [
            i for i, m in enumerate(self.metadata_store)
            if (not document_id or m.get("document_id") == document_id) and
               (not filename or m.get("filename") == filename or m.get("document_name") == filename)
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

    def clear_document(self, filename: str = None, document_id: str = None) -> None:
        keep_indices = [
            i for i, m in enumerate(self.metadata_store)
            if (filename and m.get("filename") != filename and m.get("document_name") != filename) and
               (document_id and m.get("document_id") != document_id)
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

    def _clear_internal(self):
        self.ids_store = []
        self.documents_store = []
        self.metadata_store = []
        self.vector_matrix = None
        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
        self._save_persistent_faiss()

    def clear_all(self) -> None:
        self._clear_internal()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.ids_store),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "storage_directory": str(VECTOR_DB_DIR)
        }
