import re
from typing import List, Dict, Any

class DocumentChunker:
    """
    RAG Document Chunking Engine implementing short heading merging,
    sentence-level sliding window overlap, and header injection to guarantee
    that titles and body text paragraphs are never separated into isolated chunks.
    """

    def create_chunks(
        self,
        pages_data: List[Dict[str, Any]],
        filename: str,
        strategy: str = "recursive",
        chunk_size: int = 800,
        chunk_overlap: int = 150
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
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for p in pages_data:
            page_num = p["page_number"]
            text = p["text"]
            if not text:
                continue

            raw_paragraphs = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]
            
            # Merge short headings (< 120 chars) with subsequent body text paragraphs
            merged_paragraphs = []
            buffer = ""

            for para in raw_paragraphs:
                if buffer:
                    buffer += "\n\n" + para
                    if len(buffer) >= 120:
                        merged_paragraphs.append(buffer)
                        buffer = ""
                else:
                    if len(para) < 120:
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

                if len(para_clean) <= chunk_size:
                    chunk_text = f"[Document: {filename} | Page {page_num}]\n{para_clean}"
                    chunks.append({
                        "chunk_id": f"{filename}_chunk_{chunk_index}",
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "raw_content": para_clean,
                        "filename": filename,
                        "page_number": page_num,
                        "strategy": strategy,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })
                    chunk_index += 1
                else:
                    sub_chunks = DocumentChunker._sub_split_text(para_clean, chunk_size, chunk_overlap)
                    for sc in sub_chunks:
                        chunk_text = f"[Document: {filename} | Page {page_num}]\n{sc}"
                        chunks.append({
                            "chunk_id": f"{filename}_chunk_{chunk_index}",
                            "chunk_index": chunk_index,
                            "text": chunk_text,
                            "raw_content": sc,
                            "filename": filename,
                            "page_number": page_num,
                            "strategy": strategy,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split())
                        })
                        chunk_index += 1

        return chunks

    @staticmethod
    def _sub_split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits text into sentences while enforcing sliding window sentence overlap.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        curr_sentences = []
        curr_len = 0

        for s in sentences:
            if curr_len + len(s) + 1 <= chunk_size:
                curr_sentences.append(s)
                curr_len += len(s) + 1
            else:
                if curr_sentences:
                    result.append(" ".join(curr_sentences))

                # Sliding window overlap
                overlap_sentences = []
                overlap_len = 0
                for prev_s in reversed(curr_sentences):
                    if overlap_len + len(prev_s) + 1 <= overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s) + 1
                    else:
                        break

                curr_sentences = overlap_sentences + [s]
                curr_len = sum(len(x) + 1 for x in curr_sentences)

        if curr_sentences:
            result.append(" ".join(curr_sentences))

        return result

TextChunker = DocumentChunker
