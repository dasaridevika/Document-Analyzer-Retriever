import fitz  # PyMuPDF
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    """PDF parsing service powered by PyMuPDF (fitz)"""

    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Parses raw PDF bytes into page-level structured text and metadata.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        full_text = ""
        total_word_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            # Extract plain text with layout preservation
            text = page.get_text("text") or ""
            
            # Clean up trailing whitespaces & control characters
            cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            
            word_count = len(cleaned_text.split())
            total_word_count += word_count

            pages_data.append({
                "page_number": page_num + 1,
                "text": cleaned_text,
                "word_count": word_count,
                "char_count": len(cleaned_text)
            })

            full_text += f"\n--- Page {page_num + 1} ---\n" + cleaned_text

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

        return {
            "metadata": doc_metadata,
            "pages": pages_data,
            "full_text": full_text
        }
