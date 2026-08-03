import re
from typing import List, Dict, Any

class DocumentChunker:
    """
    RAG Document Chunking Engine implementing multiple strategies:
    - Fixed Size with Overlap
    - Recursive Character Chunking
    - Page-Aware Chunking
    - Semantic Paragraph Chunking
    """

    def create_chunks(
        self,
        pages_data: List[Dict[str, Any]],
        filename: str,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        return self.chunk_document(
            pages_data=pages_data,
            filename=filename,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @staticmethod
    def chunk_document(
        pages_data: List[Dict[str, Any]],
        filename: str,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        if strategy == "fixed":
            return DocumentChunker._fixed_size_chunking(pages_data, filename, chunk_size, chunk_overlap)
        elif strategy == "page_aware":
            return DocumentChunker._page_aware_chunking(pages_data, filename, chunk_size)
        elif strategy == "semantic":
            return DocumentChunker._semantic_paragraph_chunking(pages_data, filename, chunk_size)
        else:  # Default: recursive
            return DocumentChunker._recursive_character_chunking(pages_data, filename, chunk_size, chunk_overlap)

    @staticmethod
    def _fixed_size_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        chunks = []
        full_text_pages = []
        for p in pages_data:
            full_text_pages.append((p["page_number"], p["text"]))

        chunk_index = 0
        for page_num, text in full_text_pages:
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id": f"{filename}_fixed_{chunk_index}",
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "filename": filename,
                        "page_number": page_num,
                        "strategy": "fixed",
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })
                    chunk_index += 1

                start += (chunk_size - chunk_overlap)
                if start >= text_len or chunk_size <= chunk_overlap:
                    break

        return chunks

    @staticmethod
    def _recursive_character_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        target_size: int = 500,
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for p in pages_data:
            page_num = p["page_number"]
            text = p["text"]
            if not text:
                continue

            paragraphs = re.split(r'\n\n+', text)
            current_chunk = ""

            for para in paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue

                if len(current_chunk) + len(para_clean) + 2 <= target_size:
                    current_chunk += ("\n\n" + para_clean if current_chunk else para_clean)
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"{filename}_rec_{chunk_index}",
                            "chunk_index": chunk_index,
                            "text": current_chunk,
                            "filename": filename,
                            "page_number": page_num,
                            "strategy": "recursive",
                            "char_count": len(current_chunk),
                            "word_count": len(current_chunk.split())
                        })
                        chunk_index += 1

                    if len(para_clean) > target_size:
                        sub_chunks = DocumentChunker._sub_split_text(para_clean, target_size, overlap)
                        for sc in sub_chunks:
                            chunks.append({
                                "chunk_id": f"{filename}_rec_{chunk_index}",
                                "chunk_index": chunk_index,
                                "text": sc,
                                "filename": filename,
                                "page_number": page_num,
                                "strategy": "recursive",
                                "char_count": len(sc),
                                "word_count": len(sc.split())
                            })
                            chunk_index += 1
                        current_chunk = ""
                    else:
                        current_chunk = para_clean

            if current_chunk:
                chunks.append({
                    "chunk_id": f"{filename}_rec_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": current_chunk,
                    "filename": filename,
                    "page_number": page_num,
                    "strategy": "recursive",
                    "char_count": len(current_chunk),
                    "word_count": len(current_chunk.split())
                })
                chunk_index += 1

        return chunks

    @staticmethod
    def _sub_split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        curr = ""

        for s in sentences:
            if len(curr) + len(s) + 1 <= chunk_size:
                curr += (" " + s if curr else s)
            else:
                if curr:
                    result.append(curr)
                curr = s

        if curr:
            result.append(curr)
        return result

    @staticmethod
    def _page_aware_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        target_size: int = 500
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for p in pages_data:
            page_num = p["page_number"]
            text = p["text"]
            if not text:
                continue

            if len(text) <= target_size * 1.5:
                chunks.append({
                    "chunk_id": f"{filename}_page_{page_num}_0",
                    "chunk_index": chunk_index,
                    "text": text,
                    "filename": filename,
                    "page_number": page_num,
                    "strategy": "page_aware",
                    "char_count": len(text),
                    "word_count": len(text.split())
                })
                chunk_index += 1
            else:
                sub_chunks = DocumentChunker._sub_split_text(text, target_size, 50)
                for i, sc in enumerate(sub_chunks):
                    chunks.append({
                        "chunk_id": f"{filename}_page_{page_num}_{i}",
                        "chunk_index": chunk_index,
                        "text": sc,
                        "filename": filename,
                        "page_number": page_num,
                        "strategy": "page_aware",
                        "char_count": len(sc),
                        "word_count": len(sc.split())
                    })
                    chunk_index += 1

        return chunks

    @staticmethod
    def _semantic_paragraph_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        max_chunk_size: int = 600
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for p in pages_data:
            page_num = p["page_number"]
            text = p["text"]
            if not text:
                continue

            paragraphs = re.split(r'\n\n+', text)
            current_chunk = ""

            for para in paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue

                if len(current_chunk) + len(para_clean) + 2 <= max_chunk_size:
                    current_chunk += ("\n\n" + para_clean if current_chunk else para_clean)
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"{filename}_sem_{chunk_index}",
                            "chunk_index": chunk_index,
                            "text": current_chunk,
                            "filename": filename,
                            "page_number": page_num,
                            "strategy": "semantic",
                            "char_count": len(current_chunk),
                            "word_count": len(current_chunk.split())
                        })
                        chunk_index += 1
                    current_chunk = para_clean

            if current_chunk:
                chunks.append({
                    "chunk_id": f"{filename}_sem_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": current_chunk,
                    "filename": filename,
                    "page_number": page_num,
                    "strategy": "semantic",
                    "char_count": len(current_chunk),
                    "word_count": len(current_chunk.split())
                })
                chunk_index += 1

        return chunks

# Class Alias for Backward Compatibility
TextChunker = DocumentChunker
