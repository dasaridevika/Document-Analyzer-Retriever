import os
import chromadb
import logging
from typing import List, Dict, Any
from backend.config import VECTOR_DB_DIR

os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Persistent Vector Store wrapper over ChromaDB, saved on disk (Railway persistent volume).
    """

    def __init__(self, collection_name: str = "doc_analyser_collection"):
        try:
            self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        except Exception as e:
            logger.warning(f"PersistentClient fallback initialization: {e}")
            self.client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        if not chunks or not embeddings:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "filename": c["filename"],
                "page_number": int(c["page_number"]),
                "chunk_index": int(c["chunk_index"]),
                "strategy": c["strategy"],
                "char_count": int(c["char_count"]),
                "word_count": int(c["word_count"])
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully added {len(chunks)} chunks to vector store collection '{self.collection_name}'.")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filename_filter: str = None
    ) -> List[Dict[str, Any]]:
        where_clause = {"filename": filename_filter} if filename_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                dist = distances[i]
                similarity = round(max(0.0, 1.0 - float(dist)), 4)

                retrieved_chunks.append({
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "similarity_score": similarity,
                    "distance": dist
                })

        return retrieved_chunks

    def clear_document(self, filename: str) -> None:
        try:
            self.collection.delete(where={"filename": filename})
            logger.info(f"Deleted chunks for document '{filename}' from vector store.")
        except Exception as e:
            logger.warning(f"Error clearing document '{filename}': {e}")

    def clear_all(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "storage_directory": str(VECTOR_DB_DIR)
        }
