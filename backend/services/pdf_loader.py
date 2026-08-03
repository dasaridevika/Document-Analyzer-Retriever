import logging
from typing import List, Dict, Any
from backend.services.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

class PDFLoader:
    """
    PDFLoader compatibility wrapper delegating to PDFParser.
    """

    def __init__(self):
        self.parser = PDFParser()

    def extract_pages(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        result = self.parser.parse_pdf_bytes(pdf_bytes, filename)
        return result.get("pages", [])

    def extract_metadata(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        result = self.parser.parse_pdf_bytes(pdf_bytes, filename)
        return result.get("metadata", {})
