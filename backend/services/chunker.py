import re
import uuid
import datetime
from typing import List, Dict, Any, Optional
from backend.config import DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS

try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None

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

            for para in merged_paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue

                tokens_in_para = count_tokens(para_clean)

                if tokens_in_para <= chunk_size:
                    sub_chunks_raw = [para_clean]
                else:
                    sub_chunks_raw = DocumentChunker._sub_split_by_tokens(para_clean, chunk_size, chunk_overlap)

                for sc in sub_chunks_raw:
                    sc_clean = sc.strip()
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

TextChunker = DocumentChunker
