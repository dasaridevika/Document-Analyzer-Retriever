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

    @staticmethod
    def chunk_document(
        pages_data: List[Dict[str, Any]],
        filename: str,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Main chunking dispatcher function.
        """
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
        full_text_with_pages = []

        for p in pages_data:
            full_text_with_pages.append((p["page_number"], p["text"]))

        combined_text = ""
        page_mappings = []

        for page_num, text in full_text_with_pages:
            start_pos = len(combined_text)
            combined_text += text + "\n\n"
            end_pos = len(combined_text)
            page_mappings.append((start_pos, end_pos, page_num))

        start = 0
        chunk_index = 0
        text_len = len(combined_text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = combined_text[start:end].strip()

            if chunk_text:
                # Find associated page number
                assoc_pages = [
                    p_num for p_start, p_end, p_num in page_mappings
                    if not (end <= p_start or start >= p_end)
                ]
                primary_page = assoc_pages[0] if assoc_pages else 1

                chunks.append({
                    "chunk_id": f"{filename}_fixed_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "filename": filename,
                    "page_number": primary_page,
                    "strategy": "fixed",
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                })
                chunk_index += 1

            start += chunk_size - chunk_overlap
            if start >= text_len - chunk_overlap and end == text_len:
                break

        return chunks

    @staticmethod
    def _recursive_character_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] = None
    ) -> List[Dict[str, Any]]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

        chunks = []
        chunk_index = 0

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]

            page_chunks = DocumentChunker._split_text_recursively(
                page_text, separators, chunk_size, chunk_overlap
            )

            for chunk_txt in page_chunks:
                clean_txt = chunk_txt.strip()
                if clean_txt:
                    chunks.append({
                        "chunk_id": f"{filename}_rec_{chunk_index}",
                        "chunk_index": chunk_index,
                        "text": clean_txt,
                        "filename": filename,
                        "page_number": page_num,
                        "strategy": "recursive",
                        "char_count": len(clean_txt),
                        "word_count": len(clean_txt.split())
                    })
                    chunk_index += 1

        return chunks

    @staticmethod
    def _split_text_recursively(
        text: str, separators: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        final_chunks = []
        
        # Choose separator
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        current_doc = []
        total_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if separator != "" else 0)
            if total_len + s_len > chunk_size and current_doc:
                doc_str = (separator if separator != "" else "").join(current_doc)
                if len(doc_str) > chunk_size and new_separators:
                    # Recursively split large chunk
                    sub_chunks = DocumentChunker._split_text_recursively(
                        doc_str, new_separators, chunk_size, chunk_overlap
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(doc_str)

                # Overlap logic
                overlap_doc = []
                overlap_len = 0
                for item in reversed(current_doc):
                    if overlap_len + len(item) <= chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_doc = overlap_doc
                total_len = overlap_len

            current_doc.append(s)
            total_len += s_len

        if current_doc:
            doc_str = (separator if separator != "" else "").join(current_doc)
            final_chunks.append(doc_str)

        return final_chunks

    @staticmethod
    def _page_aware_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        chunk_size: int
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for page in pages_data:
            page_num = page["page_number"]
            text = page["text"]

            if len(text) <= chunk_size:
                chunks.append({
                    "chunk_id": f"{filename}_page_{chunk_index}",
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
                # Sub-split long page
                sub_chunks = DocumentChunker._fixed_size_chunking(
                    [page], filename, chunk_size, chunk_overlap=50
                )
                for sc in sub_chunks:
                    sc["strategy"] = "page_aware"
                    sc["chunk_id"] = f"{filename}_page_{chunk_index}"
                    sc["chunk_index"] = chunk_index
                    chunks.append(sc)
                    chunk_index += 1

        return chunks

    @staticmethod
    def _semantic_paragraph_chunking(
        pages_data: List[Dict[str, Any]],
        filename: str,
        max_chunk_size: int
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for page in pages_data:
            page_num = page["page_number"]
            paragraphs = re.split(r'\n\s*\n', page["text"])

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
