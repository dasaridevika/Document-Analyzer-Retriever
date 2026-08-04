import fitz  # PyMuPDF
import re
import logging
import io
from typing import List, Dict, Any, Optional
from backend.config import MAX_PDF_PAGE_COUNT

logger = logging.getLogger(__name__)

# Try importing pytesseract / PIL for OCR fallback if installed
try:
    import pytesseract
    from PIL import Image
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

class PDFParser:
    """
    Production PDF Parser with Multi-Column Reading Order, PyMuPDF Table Extraction,
    Character Density Analysis, and OCR Fallback for Scanned Pages.
    """

    OCR_MIN_CHARS_PER_PAGE = 80
    OCR_DPI = 200

    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        pages_data = []
        warnings = []
        scanned_pages = []
        ocr_pages = []
        searchable_pages_count = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Multi-Column Preserving Text Extraction
            raw_text = page.get_text("text", sort=True)
            clean_text = PDFParser._clean_page_text(raw_text)
            char_count = len(clean_text)
            word_count = len(clean_text.split())
            extraction_method = "pymupdf"

            # 2. PyMuPDF Table Extraction
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    table_mds = []
                    for t in tables.tables:
                        df_md = t.to_markdown()
                        if df_md:
                            table_mds.append(df_md)
                    if table_mds:
                        clean_text += "\n\n### Extracted Tables:\n" + "\n\n".join(table_mds)
            except Exception as e:
                logger.debug(f"Table extraction notice on page {page_num}: {e}")

            # 3. Check for Scanned Page & OCR Fallback
            if char_count < PDFParser.OCR_MIN_CHARS_PER_PAGE:
                scanned_pages.append(page_num)

                # Attempt OCR if pytesseract is installed
                ocr_text = PDFParser._run_ocr_fallback(page)
                if ocr_text and len(ocr_text.strip()) > char_count:
                    clean_text = PDFParser._clean_page_text(ocr_text)
                    char_count = len(clean_text)
                    word_count = len(clean_text.split())
                    extraction_method = "ocr"
                    ocr_pages.append(page_num)

            is_searchable = char_count >= 20

            if is_searchable:
                searchable_pages_count += 1
                pages_data.append({
                    "page_number": page_num,
                    "text": clean_text,
                    "extraction_method": extraction_method,
                    "character_count": char_count,
                    "word_count": word_count,
                    "is_searchable": True
                })

        doc.close()

        # Generate Warnings & Extraction Report
        unsearchable = [p for p in range(1, total_pages + 1) if p not in [pd["page_number"] for pd in pages_data]]
        if unsearchable:
            if HAS_PYTESSERACT:
                warnings.append(f"Pages {unsearchable} could not be indexed despite OCR attempt.")
            else:
                warnings.append(f"Pages {unsearchable} appear to contain scanned images or minimal text. Please enable OCR or upload a text-searchable PDF.")

        quality_score = round((searchable_pages_count / max(1, total_pages)) * 100, 1)

        extraction_report = {
            "total_pages": total_pages,
            "searchable_pages": searchable_pages_count,
            "scanned_pages_detected": scanned_pages,
            "ocr_pages_processed": ocr_pages,
            "missing_unsearchable_pages": unsearchable,
            "quality_score": quality_score,
            "extraction_warnings": warnings
        }

        return {
            "filename": filename,
            "pages": pages_data,
            "extraction_report": extraction_report
        }

    @staticmethod
    def _run_ocr_fallback(page: fitz.Page) -> Optional[str]:
        """
        Renders PyMuPDF page to image pixmap and runs pytesseract OCR if available.
        """
        if not HAS_PYTESSERACT:
            return None

        try:
            pix = page.get_pixmap(dpi=PDFParser.OCR_DPI)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(image)
            return ocr_text
        except Exception as e:
            logger.warning(f"PyTesseract OCR failed on page {page.number + 1}: {e}")
            return None

    @staticmethod
    def _clean_page_text(text: str) -> str:
        if not text:
            return ""
        # Fix hyphens at end of line
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Normalize repetitive spaces & control characters
        text = re.sub(r'[\r\t]', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
