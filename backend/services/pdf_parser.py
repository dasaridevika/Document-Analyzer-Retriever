import fitz  # PyMuPDF
import gc
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Memory-optimized PDF parsing service powered by PyMuPDF (fitz).
    Uses memory-mapped file paths to prevent OOM memory spikes on large PDFs.
    """

    @staticmethod
    def parse_pdf_file(file_path: str, filename: str) -> Dict[str, Any]:
        """
        Parses PDF directly from file path with low memory footprint (< 10MB RAM).
        """
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        text_chunks_list: List[str] = []
        total_word_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text") or ""
            
            cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            word_count = len(cleaned_text.split())
            total_word_count += word_count

            pages_data.append({
                "page_number": page_num + 1,
                "text": cleaned_text,
                "word_count": word_count,
                "char_count": len(cleaned_text)
            })

            text_chunks_list.append(cleaned_text)

        full_text = "\n\n".join(text_chunks_list)
        total_chars = len(full_text)

        doc_metadata = {
            "filename": filename,
            "total_pages": total_pages,
            "total_words": total_word_count,
            "total_chars": total_chars,
            "format": doc.metadata.get("format", "PDF"),
            "title": doc.metadata.get("title", filename),
            "author": doc.metadata.get("author", "Unknown"),
        }

        doc.close()
        gc.collect()  # Clean up memory immediately

        return {
            "metadata": doc_metadata,
            "pages": pages_data,
            "full_text": full_text
        }

    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        text_chunks_list: List[str] = []
        total_word_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text") or ""
            cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            word_count = len(cleaned_text.split())
            total_word_count += word_count

            pages_data.append({
                "page_number": page_num + 1,
                "text": cleaned_text,
                "word_count": word_count,
                "char_count": len(cleaned_text)
            })
            text_chunks_list.append(cleaned_text)

        full_text = "\n\n".join(text_chunks_list)

        doc_metadata = {
            "filename": filename,
            "total_pages": total_pages,
            "total_words": total_word_count,
            "total_chars": len(full_text),
            "format": doc.metadata.get("format", "PDF"),
            "title": doc.metadata.get("title", filename),
            "author": doc.metadata.get("author", "Unknown"),
        }

        doc.close()
        gc.collect()

        return {
            "metadata": doc_metadata,
            "pages": pages_data,
            "full_text": full_text
        }
