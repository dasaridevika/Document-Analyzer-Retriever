import fitz  # PyMuPDF
import gc
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    """
    High-Fidelity Reading-Order PDF Parsing Service powered by PyMuPDF (fitz).
    Extracts text block-by-block with physical sort=True to preserve tables, headings, and lists cleanly.
    """

    @staticmethod
    def _extract_page_text(page) -> str:
        """
        Extracts page text in exact physical reading order using sorted text blocks.
        """
        try:
            blocks = page.get_text("blocks", sort=True)
            text_parts = []
            for b in blocks:
                # b[4] is block text, b[6] is block type (0 for text)
                if len(b) >= 5 and b[6] == 0:
                    b_text = b[4].strip()
                    if b_text:
                        text_parts.append(b_text)
            
            if text_parts:
                return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Blocks extraction failed: {e}. Falling back to default text sort.")

        raw_text = page.get_text("text", sort=True) or ""
        return "\n\n".join([line.strip() for line in raw_text.splitlines() if line.strip()])

    @staticmethod
    def parse_pdf_file(file_path: str, filename: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        text_chunks_list: List[str] = []
        total_word_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            cleaned_text = PDFParser._extract_page_text(page)
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

    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        text_chunks_list: List[str] = []
        total_word_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            cleaned_text = PDFParser._extract_page_text(page)
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
