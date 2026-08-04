import re
import uuid
import datetime
import logging
from typing import List, Dict, Any, Optional
from backend.config import DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS

try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None

logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    if encoding:
        return len(encoding.encode(text))
    return max(1, len(text) // 4)

def validate_index_coverage(pdf_page_count: int, chunks: List[Dict[str, Any]]) -> List[int]:
    """
    Step 3 Requirement: Validates which pages have indexed chunks and returns sorted list of missing page numbers.
    """
    indexed_pages = {chunk["page_number"] for chunk in chunks if "page_number" in chunk}
    missing_pages = set(range(1, pdf_page_count + 1)) - indexed_pages
    return sorted(missing_pages)


class DocumentChunker:
    """
    Production Token-Aware Contextual Chunking Engine.
    Supports recursive paragraph chunking and semantic shift chunking.
    - Preserves extraction_method (pymupdf / ocr) per chunk
    - Index coverage validation
    """

    def create_chunks(
        self,
        pages_data: List[Dict[str, Any]],
        filename: str,
        document_id: str = "",
        session_id: str = "",
        user_id: str = "anonymous_user",
        strategy: str = "recursive",
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        embedding_model: str = "@cf/baai/bge-large-en-v1.5",
        embedding_dim: int = 1024
    ) -> List[Dict[str, Any]]:
        return self.chunk_document(
            pages_data=pages_data,
            filename=filename,
            document_id=document_id,
            session_id=session_id,
            user_id=user_id,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim
        )

    @staticmethod
    def _extract_section_title(text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:3]:
            if line.isupper() or re.match(r'^(?:#|\d+\.|\bSection\b|\bChapter\b|\bUnit\b|\bFeature\b)', line, re.IGNORECASE):
                clean_t = re.sub(r'^[#\d\.\s]+', '', line).strip()
                if 3 <= len(clean_t) <= 80:
                    return clean_t
        return "General Section"

    @staticmethod
    def _split_semantic(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        # Split on sentence boundaries using fixed-width lookbehind
        raw_splits = re.split(r'(?<=[.!?])\s+|\n', text)
        sentences = []
        abbreviations = {"e.g.", "i.e.", "dr.", "st.", "mr.", "mrs.", "ms.", "prof.", "vs."}
        
        buffer = []
        for segment in raw_splits:
            s_clean = segment.strip()
            if not s_clean:
                continue
            buffer.append(s_clean)
            
            # Check if this segment ends with a common abbreviation
            words = s_clean.split()
            last_word = words[-1].lower() if words else ""
            if last_word in abbreviations:
                continue
            else:
                sentences.append(" ".join(buffer))
                buffer = []
        if buffer:
            sentences.append(" ".join(buffer))

        if len(sentences) < 2:
            return [text]

        # Dynamically load embedding service to avoid circular dependency
        try:
            from backend.services.embeddings import EmbeddingService
            embedder = EmbeddingService()
            embeddings = embedder.generate_embeddings(sentences)
        except Exception as e:
            logger.warning(f"EmbeddingService failed in semantic split: {e}. Falling back to standard sentence token splitting.")
            return DocumentChunker._sub_split_by_tokens(text, chunk_size, chunk_overlap)

        if not embeddings or len(embeddings) != len(sentences):
            return DocumentChunker._sub_split_by_tokens(text, chunk_size, chunk_overlap)

        # Calculate cosine similarities
        import numpy as np
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i+1])
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        distances = [1.0 - sim for sim in similarities]
        
        # Split on distance jumps (70th percentile threshold)
        if distances:
            threshold = np.percentile(distances, 70)
            split_indices = [i for i, d in enumerate(distances) if d >= threshold]
        else:
            split_indices = []

        # Group sentences into initial raw semantic chunks
        raw_chunks = []
        current_chunk = []
        for idx, s in enumerate(sentences):
            current_chunk.append(s)
            if idx in split_indices:
                raw_chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            raw_chunks.append(" ".join(current_chunk))

        # Merge tiny adjacent chunks to prevent over-fragmentation
        merged_chunks = []
        current_buffer = []
        current_tokens = 0

        for chunk in raw_chunks:
            chunk_tokens = count_tokens(chunk)
            if current_tokens + chunk_tokens <= chunk_size:
                current_buffer.append(chunk)
                current_tokens += chunk_tokens
            else:
                if current_buffer:
                    merged_chunks.append(" ".join(current_buffer))
                current_buffer = [chunk]
                current_tokens = chunk_tokens
        if current_buffer:
            merged_chunks.append(" ".join(current_buffer))

        # Finally, guarantee token boundaries (sub-split any chunks that are too large)
        final_splits = []
        for chunk in merged_chunks:
            tokens = count_tokens(chunk)
            if tokens <= chunk_size:
                final_splits.append(chunk)
            else:
                final_splits.extend(DocumentChunker._sub_split_by_tokens(chunk, chunk_size, chunk_overlap))

        return final_splits

    @staticmethod
    def chunk_document(
        pages_data: List[Dict[str, Any]],
        filename: str,
        document_id: str = "",
        session_id: str = "",
        user_id: str = "anonymous_user",
        strategy: str = "recursive",
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        embedding_model: str = "@cf/baai/bge-large-en-v1.5",
        embedding_dim: int = 1024
    ) -> List[Dict[str, Any]]:
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        clean_uid = user_id.strip().lower() if user_id else "anonymous_user"
        created_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        chunks = []
        chunk_index = 0
        seen_texts = set()

        for p in pages_data:
            page_num = p["page_number"]
            text = p["text"]
            extraction_method = p.get("extraction_method", "pymupdf")

            if not text:
                continue

            section_title = DocumentChunker._extract_section_title(text)

            # Choose chunking strategy
            if strategy == "semantic":
                sub_chunks_raw = DocumentChunker._split_semantic(text, chunk_size, chunk_overlap)
            elif strategy == "hierarchical":
                parents = DocumentChunker._sub_split_by_tokens(text, max_tokens=600, overlap_tokens=100)
                sub_chunks_raw = []
                for p_idx, parent in enumerate(parents):
                    pid = f"{doc_id}_p{chunk_index}_parent{p_idx}"
                    # Children chunks
                    children = DocumentChunker._sub_split_by_tokens(parent, max_tokens=150, overlap_tokens=30)
                    for child in children:
                        sub_chunks_raw.append({
                            "text": child,
                            "parent_id": pid,
                            "parent_text": parent
                        })
            else:
                raw_paragraphs = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]
                
                merged_paragraphs = []
                buffer = ""

                for para in raw_paragraphs:
                    if buffer:
                        buffer += "\n\n" + para
                        if count_tokens(buffer) >= 30:
                            merged_paragraphs.append(buffer)
                            buffer = ""
                    else:
                        if count_tokens(para) < 30:
                            buffer = para
                        else:
                            merged_paragraphs.append(para)

                if buffer:
                    if merged_paragraphs:
                        merged_paragraphs[-1] += "\n\n" + buffer
                    else:
                        merged_paragraphs.append(buffer)

                sub_chunks_raw = []
                for para in merged_paragraphs:
                    para_clean = para.strip()
                    if not para_clean:
                        continue

                    tokens_in_para = count_tokens(para_clean)

                    if tokens_in_para <= chunk_size:
                        sub_chunks_raw.append(para_clean)
                    else:
                        sub_chunks_raw.extend(DocumentChunker._sub_split_by_tokens(para_clean, chunk_size, chunk_overlap))

            for sc in sub_chunks_raw:
                if isinstance(sc, dict):
                    sc_clean = sc["text"].strip()
                    parent_text = sc["parent_text"]
                    parent_id = sc["parent_id"]
                else:
                    sc_clean = sc.strip()
                    parent_text = ""
                    parent_id = ""

                if not sc_clean or sc_clean in seen_texts:
                    continue
                seen_texts.add(sc_clean)

                cid = f"{doc_id}_c{chunk_index}"

                formatted_chunk_text = (
                    f"Document: {filename}\n"
                    f"Document ID: {doc_id}\n"
                    f"Page: {page_num}\n"
                    f"Section: {section_title}\n"
                    f"Chunk ID: {cid}\n\n"
                    f"Content:\n{sc_clean}"
                )

                chunks.append({
                    "chunk_id": cid,
                    "chunk_index": chunk_index,
                    "document_id": doc_id,
                    "session_id": sess_id,
                    "user_id": clean_uid,
                    "filename": filename,
                    "document_name": filename,
                    "document_version": "1.0",
                    "page_number": page_num,
                    "section_title": section_title,
                    "text": formatted_chunk_text,
                    "raw_content": sc_clean,
                    "parent_text": parent_text,
                    "parent_chunk_id": parent_id,
                    "strategy": strategy,
                    "extraction_method": extraction_method,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dim,
                    "created_at": created_timestamp,
                    "token_count": count_tokens(formatted_chunk_text),
                    "char_count": len(formatted_chunk_text)
                })
                chunk_index += 1

        return chunks

    @staticmethod
    def _sub_split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+|\n', text)
        result = []
        curr_sentences = []
        curr_tokens = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_t = count_tokens(s_clean)

            if curr_tokens + s_t + 1 <= max_tokens:
                curr_sentences.append(s_clean)
                curr_tokens += s_t + 1
            else:
                if curr_sentences:
                    result.append(" ".join(curr_sentences))

                overlap_sentences = []
                overlap_t = 0
                for prev_s in reversed(curr_sentences):
                    prev_t = count_tokens(prev_s)
                    if overlap_t + prev_t + 1 <= overlap_tokens:
                        overlap_sentences.insert(0, prev_s)
                        overlap_t += prev_t + 1
                    else:
                        break

                curr_sentences = overlap_sentences + [s_clean]
                curr_tokens = sum(count_tokens(x) + 1 for x in curr_sentences)

        if curr_sentences:
            result.append(" ".join(curr_sentences))

        return result

# Export aliases
TextChunker = DocumentChunker
