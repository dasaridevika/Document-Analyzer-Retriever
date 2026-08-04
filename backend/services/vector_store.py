import os
import json
import logging
import math
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from backend.config import VECTOR_DB_DIR, MAX_CHUNKS_PER_PAGE, DENSE_TOP_K, KEYWORD_TOP_K

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
FAISS_META_PATH = VECTOR_DB_DIR / "faiss_metadata.json"


class PersistentBM25Index:
    """
    Persisted BM25 Index containing global document frequency mappings.
    Replaces query-time index compilation.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}  # chunk_id -> length
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}  # chunk_id -> term -> frequency
        self.global_df: Dict[str, int] = {}  # term -> df
        self.total_chunks = 0
        self.avg_doc_len = 0.0

    def add_document(self, chunk_id: str, text: str):
        words = text.lower().split()
        doc_len = len(words)
        self.doc_lengths[chunk_id] = doc_len
        self.total_chunks += 1
        
        # Calculate term frequencies
        tf: Dict[str, int] = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        self.doc_term_freqs[chunk_id] = tf
        
        # Update global document frequencies
        for word in tf.keys():
            self.global_df[word] = self.global_df.get(word, 0) + 1
            
        # Recalculate average length
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, self.total_chunks)

    def score_candidates(self, query: str, candidate_ids: List[str]) -> Dict[str, float]:
        q_words = query.lower().split()
        scores: Dict[str, float] = {cid: 0.0 for cid in candidate_ids}
        
        for qw in q_words:
            df = self.global_df.get(qw, 0)
            if df == 0:
                continue
            # Calculate IDF
            idf = math.log((self.total_chunks - df + 0.5) / (df + 0.5) + 1.0)
            
            for cid in candidate_ids:
                tf = self.doc_term_freqs.get(cid, {}).get(qw, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths.get(cid, 0)
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                scores[cid] += idf * (num / denom)
                
        return scores

    def save_state(self) -> Dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_lengths": self.doc_lengths,
            "doc_term_freqs": self.doc_term_freqs,
            "global_df": self.global_df,
            "total_chunks": self.total_chunks,
            "avg_doc_len": self.avg_doc_len
        }

    def load_state(self, state: Dict[str, Any]):
        self.k1 = state.get("k1", 1.5)
        self.b = state.get("b", 0.75)
        self.doc_lengths = state.get("doc_lengths", {})
        self.doc_term_freqs = state.get("doc_term_freqs", {})
        self.global_df = state.get("global_df", {})
        self.total_chunks = state.get("total_chunks", 0)
        self.avg_doc_len = state.get("avg_doc_len", 0.0)


def _jaccard_similarity(text1: str, text2: str) -> float:
    w1 = set(re.findall(r'\w+', text1.lower()))
    w2 = set(re.findall(r'\w+', text2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def maximum_marginal_relevance(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 8
) -> List[int]:
    """
    Computes Maximum Marginal Relevance (MMR) to balance relevance with diversity.
    """
    if len(doc_vectors) == 0:
        return []

    # Relevance to query
    relevance = np.dot(doc_vectors, query_vector.T).flatten()
    selected_indices: List[int] = []
    unselected_indices = list(range(len(doc_vectors)))

    while len(selected_indices) < min(top_k, len(doc_vectors)):
        best_idx = -1
        best_mmr = -1e9

        for idx in unselected_indices:
            rel_score = relevance[idx]
            if not selected_indices:
                redundancy = 0.0
            else:
                selected_vecs = doc_vectors[selected_indices]
                redundancy = max(np.dot(selected_vecs, doc_vectors[idx].T).flatten())

            mmr_score = lambda_param * rel_score - (1.0 - lambda_param) * redundancy
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        else:
            break

    return selected_indices


class VectorStoreManager:
    """
    Production-Grade Vector Store Engine with Hybrid Search, MMR Reranking & Jaccard Deduplication.
    """

    def __init__(self, embedding_dim: int = 1024, embedding_model: str = "@cf/baai/bge-large-en-v1.5"):
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.faiss_index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.documents_store: List[str] = []
        self.ids_store: List[str] = []
        self.vector_matrix: Optional[np.ndarray] = None
        self.bm25_index = PersistentBM25Index()

        # Inverted index dictionary maps
        self._doc_id_to_indices = defaultdict(list)
        self._filename_to_indices = defaultdict(list)
        self._session_id_to_indices = defaultdict(list)
        self._user_id_to_indices = defaultdict(list)

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
                        logger.warning(f"Embedding model mismatch detected. Initializing fresh collection.")
                        self._clear_internal()
                        return

                    self.metadata_store = meta_data.get("metadatas", [])
                    self.documents_store = meta_data.get("documents", [])
                    self.ids_store = meta_data.get("ids", [])
                    stored_vecs = meta_data.get("vectors")
                    if stored_vecs:
                        self.vector_matrix = np.array(stored_vecs, dtype=np.float32)

                    # Load BM25 state
                    bm25_state = meta_data.get("bm25_index")
                    if bm25_state:
                        self.bm25_index.load_state(bm25_state)
                    else:
                        # Rebuild if missing
                        for i, doc in enumerate(self.documents_store):
                            self.bm25_index.add_document(self.ids_store[i], doc)

                    # Rebuild inverted indices
                    self._rebuild_inverted_indices()

                logger.info(f"Loaded existing vector index with {len(self.ids_store)} chunks.")
                return
            except Exception as e:
                logger.warning(f"Error loading vector index: {e}. Initializing clean index.")

        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)

    def _rebuild_inverted_indices(self):
        self._doc_id_to_indices = defaultdict(list)
        self._filename_to_indices = defaultdict(list)
        self._session_id_to_indices = defaultdict(list)
        self._user_id_to_indices = defaultdict(list)
        
        for i, meta in enumerate(self.metadata_store):
            d_id = meta.get("document_id")
            if d_id:
                self._doc_id_to_indices[d_id].append(i)
            fname = meta.get("filename") or meta.get("document_name")
            if fname:
                self._filename_to_indices[fname].append(i)
            s_id = meta.get("session_id")
            if s_id:
                self._session_id_to_indices[s_id].append(i)
            u_id = meta.get("user_id")
            if u_id:
                self._user_id_to_indices[u_id].append(i)

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
                "vectors": self.vector_matrix.tolist() if self.vector_matrix is not None else [],
                "bm25_index": self.bm25_index.save_state()
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
            self.bm25_index.add_document(c["chunk_id"], c["text"])
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
                "parent_text": c.get("parent_text", ""),
                "parent_chunk_id": c.get("parent_chunk_id", ""),
                "strategy": c.get("strategy", "recursive"),
                "extraction_method": c.get("extraction_method", "PyMuPDF Reading Order"),
                "embedding_model": c.get("embedding_model", self.embedding_model),
                "embedding_dimension": int(c.get("embedding_dimension", self.embedding_dim)),
                "created_at": c.get("created_at", "")
            })

        self._rebuild_inverted_indices()
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
        intent_type: str = "fact",
        top_k: int = 12,
        filename_filter: str = None,
        document_id_filter: str = None,
        session_id_filter: str = None,
        user_id_filter: str = None,
        min_score: float = 0.15,
        use_mmr: bool = True
    ) -> List[Dict[str, Any]]:
        if not self.ids_store or self.vector_matrix is None:
            return []

        norm_query = self._normalize_vectors([query_embedding])

        if norm_query.shape[1] != self.vector_matrix.shape[1]:
            logger.warning(f"Embedding dimension mismatch: Query ({norm_query.shape[1]}) vs Store ({self.vector_matrix.shape[1]})")
            return []

        # Inverted index lookup for O(1) metadata filtering
        candidates = set(range(len(self.ids_store)))
        
        if document_id_filter:
            candidates &= set(self._doc_id_to_indices.get(document_id_filter, []))
        if filename_filter:
            candidates &= set(self._filename_to_indices.get(filename_filter, []))
        if session_id_filter:
            candidates &= set(self._session_id_to_indices.get(session_id_filter, []))
        if user_id_filter:
            candidates &= set(self._user_id_to_indices.get(user_id_filter, []))
            
        candidate_indices = sorted(list(candidates))

        if not candidate_indices:
            return []

        sub_matrix = self.vector_matrix[candidate_indices]
        sub_docs = [self.documents_store[i] for i in candidate_indices]

        # 1. Dense Cosine Similarity
        dense_sims = np.dot(sub_matrix, norm_query.T).flatten()
        dense_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(dense_sims)[::-1])}

        # 2. Lexical BM25 Search (Using Persisted Global TF/DF Statistics)
        bm25_rank_map = {}
        if raw_query.strip() and candidate_indices:
            candidate_ids = [self.ids_store[i] for i in candidate_indices]
            bm25_scores_dict = self.bm25_index.score_candidates(raw_query, candidate_ids)
            bm25_scores = np.array([bm25_scores_dict.get(self.ids_store[i], 0.0) for i in candidate_indices], dtype=np.float32)
            bm25_rank_map = {loc_idx: rank for rank, loc_idx in enumerate(np.argsort(bm25_scores)[::-1])}

        # 3. Intent-Driven RRF Scoring
        rrf_scores = []
        k_rrf = 60
        for loc_idx in range(len(candidate_indices)):
            d_rank = dense_rank_map.get(loc_idx, 999)
            b_rank = bm25_rank_map.get(loc_idx, 999)
            rrf = (1.0 / (k_rrf + d_rank)) + (1.0 / (k_rrf + b_rank))

            doc_text_lower = sub_docs[loc_idx].lower()
            
            # Generalized query overlap density (de-biased)
            doc_words = set(re.findall(r'\w+', doc_text_lower))
            query_words = set(re.findall(r'\w+', raw_query.lower())) - {
                "what", "is", "the", "how", "to", "in", "of", "for", "a", "an", "and", "or", "are", "about", "explain"
            }
            overlap = len(query_words & doc_words) / max(1, len(query_words)) if query_words else 0.0
            
            # Scale intent bonus proportionally
            intent_bonus = overlap * 0.05

            final_rerank_score = rrf + intent_bonus
            rrf_scores.append((final_rerank_score, loc_idx, dense_sims[loc_idx]))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)

        # 4. Optional Maximum Marginal Relevance (MMR) Reranking for Diversity
        if use_mmr and len(rrf_scores) > top_k:
            candidate_vecs = np.array([sub_matrix[loc_idx] for _, loc_idx, _ in rrf_scores[:top_k * 2]])
            mmr_sel = maximum_marginal_relevance(norm_query[0], candidate_vecs, lambda_param=0.75, top_k=top_k)
            final_ordered = [rrf_scores[idx] for idx in mmr_sel]
        else:
            final_ordered = rrf_scores[:top_k]

        retrieved_chunks = []
        page_counts: Dict[int, int] = {}
        selected_texts = []

        for rerank_score, loc_idx, d_score in final_ordered:
            if d_score < min_score and len(retrieved_chunks) >= 3:
                continue

            orig_idx = candidate_indices[loc_idx]
            cand_text = self.documents_store[orig_idx]
            meta = self.metadata_store[orig_idx]
            p_num = meta.get("page_number", 1)

            if page_counts.get(p_num, 0) >= MAX_CHUNKS_PER_PAGE:
                continue

            if any(_jaccard_similarity(cand_text, prev_t) > 0.70 for prev_t in selected_texts):
                continue

            selected_texts.append(cand_text)
            page_counts[p_num] = page_counts.get(p_num, 0) + 1

            retrieved_chunks.append({
                "chunk_id": self.ids_store[orig_idx],
                "text": cand_text,
                "metadata": meta,
                "similarity_score": round(float(d_score), 4),
                "rrf_score": round(float(rerank_score), 4),
                "distance": round(float(1.0 - d_score), 4)
            })

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

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

        # Rebuild BM25 index for remaining documents
        self.bm25_index = PersistentBM25Index()
        for i in keep_indices:
            self.bm25_index.add_document(self.ids_store[i], self.documents_store[i])

        if self.vector_matrix is not None and len(keep_indices) > 0:
            self.vector_matrix = self.vector_matrix[keep_indices]
        else:
            self.vector_matrix = None

        if self.faiss_module and self.vector_matrix is not None:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(self.vector_matrix)

        self._rebuild_inverted_indices()
        self._save_persistent_faiss()

    def _clear_internal(self):
        self.ids_store = []
        self.documents_store = []
        self.metadata_store = []
        self.vector_matrix = None
        self.bm25_index = PersistentBM25Index()
        if self.faiss_module:
            self.faiss_index = self.faiss_module.IndexFlatIP(self.embedding_dim)
        self._rebuild_inverted_indices()
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
