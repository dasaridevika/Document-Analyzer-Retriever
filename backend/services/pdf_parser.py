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

            # 1. Check for Tables and Text Blocks, Sorting and Inlining Tables
            tables_list = []
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    tables_list = tables.tables
            except Exception as e:
                logger.debug(f"Table search notice on page {page_num}: {e}")

            blocks = page.get_text("blocks")
            elements = []

            # Add tables to elements list
            for t in tables_list:
                tb_md = t.to_markdown()
                if tb_md:
                    elements.append((t.bbox[1], t.bbox[0], "table", tb_md))

            # Add non-table text blocks to elements list
            for b in blocks:
                bx0, by0, bx1, by1, b_text, b_no, b_type = b
                if b_type == 0:  # Text block
                    if PDFParser._is_inside_table((bx0, by0, bx1, by1), tables_list):
                        continue
                    clean_b_text = b_text.strip()
                    if clean_b_text:
                        elements.append((by0, bx0, "text", clean_b_text))

            # Sort all elements (tables and text blocks) by y0 (vertical), then x0 (horizontal)
            elements.sort(key=lambda x: (x[0], x[1]))

            # Assemble page text
            assembled_parts = []
            for el in elements:
                if el[2] == "table":
                    assembled_parts.append(f"\n\n{el[3]}\n\n")
                else:
                    assembled_parts.append(el[3])

            raw_text = "\n".join(assembled_parts)
            clean_text = PDFParser._clean_page_text(raw_text)
            char_count = len(clean_text)
            word_count = len(clean_text.split())
            extraction_method = "pymupdf"

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

    @staticmethod
    def _is_inside_table(block_bbox: tuple, tables: list) -> bool:
        bx0, by0, bx1, by1 = block_bbox
        block_area = (bx1 - bx0) * (by1 - by0)
        if block_area <= 0:
            return False
        for t in tables:
            tx0, ty0, tx1, ty1 = t.bbox
            # Intersection coordinates
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)
            if ix0 < ix1 and iy0 < iy1:
                intersect_area = (ix1 - ix0) * (iy1 - iy0)
                if intersect_area / block_area > 0.5:
                    return True
        return False
