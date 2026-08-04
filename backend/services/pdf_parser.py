import fitz  # PyMuPDF
import gc
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Production PDF Parsing Service powered by PyMuPDF (fitz).
    - Multi-column physical reading order (sort=True)
    - Table detection and markdown conversion
    - Repeating header/footer noise stripping
    - Detailed Extraction Quality Report (detects empty/scanned pages)
    """

    @staticmethod
    def _clean_text_noise(text: str) -> str:
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if re.search(r'(weebly\.com|download now|http[s]?://|www\.|\.org|\.edu)', l, re.IGNORECASE):
                continue
            if re.search(r'(collected\s*&\s*prepared\s*by|assoc\.\s*prof|m\.tech|miste|b\.tech,\s*eee|iv\s*year\s*i\s*sem|gcet,\s*kadapa)', l, re.IGNORECASE):
                continue
            if re.match(r'^\d{1,3}$', l):
                continue
            clean_lines.append(l)

        return "\n".join(clean_lines)

    @staticmethod
    def _extract_tables_markdown(page) -> Tuple[str, int]:
        """
        Extracts structured tables using PyMuPDF find_tables() if available.
        """
        table_markdowns = []
        count = 0
        try:
            if hasattr(page, "find_tables"):
                tabs = page.find_tables()
                for tab in tabs:
                    df = tab.extract()
                    if df and len(df) > 1:
                        count += 1
                        header = df[0]
                        rows = df[1:]
                        clean_header = [str(c or "").strip().replace("\n", " ") for c in header]
                        md_lines = ["| " + " | ".join(clean_header) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(clean_header)) + " |")
                        for r in rows:
                            clean_r = [str(c or "").strip().replace("\n", " ") for c in r]
                            md_lines.append("| " + " | ".join(clean_r) + " |")
                        table_markdowns.append("\n".join(md_lines))
        except Exception as e:
            logger.debug(f"Table extraction notice: {e}")
        
        return ("\n\n".join(table_markdowns), count)

    @staticmethod
    def _extract_page_content(page) -> Tuple[str, int, bool]:
        """
        Extracts text blocks and tables from a single page.
        Returns: (cleaned_page_text, table_count, is_scanned_or_empty)
        """
        table_md, table_count = PDFParser._extract_tables_markdown(page)

        try:
            blocks = page.get_text("blocks", sort=True)
            text_parts = []
            for b in blocks:
                if len(b) >= 5 and b[6] == 0:
                    b_text = b[4].strip()
                    clean_b = PDFParser._clean_text_noise(b_text)
                    if clean_b and len(clean_b) > 15:
                        text_parts.append(clean_b)
            
            page_text = "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Blocks extraction failed: {e}. Falling back to default text sort.")
            raw_text = page.get_text("text", sort=True) or ""
            page_text = PDFParser._clean_text_noise(raw_text)

        if table_md:
            page_text = (page_text + "\n\n" + table_md).strip()

        is_scanned = (len(page_text.strip()) < 30 and len(page.get_images()) > 0) or (len(page_text.strip()) < 20)

        return (page_text, table_count, is_scanned)

    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_data: List[Dict[str, Any]] = []
        text_chunks_list: List[str] = []
        total_word_count = 0
        total_char_count = 0
        empty_pages = []
        ocr_needed_pages = []
        total_tables = 0
        warnings = []

        for page_num in range(total_pages):
            page = doc[page_num]
            p_num = page_num + 1
            cleaned_text, table_count, is_scanned = PDFParser._extract_page_content(page)
            total_tables += table_count

            if is_scanned:
                ocr_needed_pages.append(p_num)
                empty_pages.append(p_num)

            if not cleaned_text or len(cleaned_text.strip()) < 20:
                continue

            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            total_word_count += word_count
            total_char_count += char_count

            pages_data.append({
                "page_number": p_num,
                "text": cleaned_text,
                "word_count": word_count,
                "char_count": char_count,
                "tables_count": table_count
            })
            text_chunks_list.append(cleaned_text)

        full_text = "\n\n".join(text_chunks_list)

        # Extraction Quality Score calculation
        quality_score = 100.0
        if total_pages > 0:
            scanned_ratio = len(ocr_needed_pages) / total_pages
            quality_score = max(0.0, round(100.0 * (1.0 - scanned_ratio), 1))

        if ocr_needed_pages:
            warnings.append(f"Pages {ocr_needed_pages} appear to contain scanned images or minimal text.")

        if total_char_count < 100 and total_pages > 0:
            warnings.append("Document extracted character count is very low. Check if PDF is image-only.")

        extraction_report = {
            "total_pages": total_pages,
            "extracted_char_count": total_char_count,
            "extracted_word_count": total_word_count,
            "empty_pages": empty_pages,
            "ocr_needed_pages": ocr_needed_pages,
            "tables_detected": total_tables,
            "quality_score": quality_score,
            "extraction_warnings": warnings
        }

        doc_metadata = {
            "filename": filename,
            "total_pages": total_pages,
            "total_words": total_word_count,
            "total_chars": total_char_count,
            "format": doc.metadata.get("format", "PDF"),
            "title": doc.metadata.get("title", filename),
            "author": doc.metadata.get("author", "Unknown"),
            "extraction_report": extraction_report
        }

        doc.close()
        gc.collect()

        return {
            "metadata": doc_metadata,
            "pages": pages_data,
            "full_text": full_text,
            "extraction_report": extraction_report
        }
