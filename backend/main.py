import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

BASE_ROOT = Path(__file__).resolve().parent.parent
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import (
    DATA_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_PDF_PAGE_COUNT,
    ALLOWED_FILE_EXTENSIONS,
    DEBUG_RAG
)
from backend.services.pdf_parser import PDFParser
from backend.services.chunker import DocumentChunker
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStoreManager
from backend.services.rag_engine import RAGEngine
from backend.services.history_store import HistoryStore
from backend.services.storage_bucket import StorageBucketManager
from backend.services.auth_firebase import FirebaseAuthService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DocAnalyzerBackend")

app = FastAPI(title="DocAnalyzer Enterprise RAG API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services Initialization
chunker = DocumentChunker()
embedding_service = EmbeddingService()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(embedding_service=embedding_service, vector_store=vector_store)
history_store = HistoryStore()
storage_bucket = StorageBucketManager()
auth_service = FirebaseAuthService()

# Request Models
class ProcessRequest(BaseModel):
    filename: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: str = "anonymous_user"
    strategy: str = "recursive"
    chunk_size: int = 400
    chunk_overlap: int = 80

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    user_id: str = "anonymous_user"
    query: str = Field(..., min_length=1)
    filename: Optional[str] = None
    system_prompt: Optional[str] = None
    top_k: int = 12
    temperature: float = 0.0

class VerifyAuthRequest(BaseModel):
    token_or_email: str

def sanitize_filename(name: str) -> str:
    clean_name = Path(name).name
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean_name)

@app.get("/api/health")
def health_check():
    stats = vector_store.get_stats()
    return {
        "status": "online",
        "service": "DocAnalyzer Enterprise RAG API",
        "version": "4.0.0",
        "vector_stats": stats,
        "debug_rag_enabled": DEBUG_RAG,
        "bucket_name": storage_bucket.bucket_name
    }

@app.post("/api/auth/verify")
def verify_auth_token(req: VerifyAuthRequest):
    token_or_email = req.token_or_email.strip()
    if not token_or_email:
        raise HTTPException(status_code=400, detail="Missing authentication token or email")

    user_info = auth_service.verify_firebase_token(token_or_email)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    profile = storage_bucket.save_user_profile(user_info["email"], user_info.get("name", ""))
    return {"status": "authenticated", "user": user_info, "profile": profile}

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous_user")
):
    safe_name = sanitize_filename(file.filename)
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Corrupted file: Missing valid PDF header signature.")

    clean_user_id = user_id.strip().lower() if user_id else "anonymous_user"
    doc_id = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"

    bucket_result = storage_bucket.save_file(
        filename=safe_name,
        content=content,
        user_id=clean_user_id,
        content_type="application/pdf"
    )

    parse_result = PDFParser.parse_pdf_bytes(content, safe_name)
    report = parse_result["extraction_report"]

    if report["total_pages"] > MAX_PDF_PAGE_COUNT:
        raise HTTPException(status_code=400, detail=f"PDF page count ({report['total_pages']}) exceeds maximum limit of {MAX_PDF_PAGE_COUNT} pages.")

    return {
        "document_id": doc_id,
        "filename": safe_name,
        "user_id": clean_user_id,
        "size_bytes": len(content),
        "bucket_info": bucket_result,
        "extraction_report": report
    }

@app.post("/api/process")
def process_document(req: ProcessRequest):
    safe_name = sanitize_filename(req.filename)
    clean_user_id = req.user_id.strip().lower() if req.user_id else "anonymous_user"

    pdf_bytes = storage_bucket.get_file(safe_name)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found in Storage Bucket.")

    doc_id = req.document_id or f"doc_{hashlib.sha256(pdf_bytes).hexdigest()[:16]}"
    sess_id = req.session_id or f"sess_{int(time.time())}"

    vector_store.clear_document(filename=safe_name, document_id=doc_id)

    parse_result = PDFParser.parse_pdf_bytes(pdf_bytes, safe_name)
    pages_data = parse_result["pages"]

    if not pages_data:
        raise HTTPException(status_code=400, detail="Failed to extract readable text pages from PDF.")

    chunks = chunker.create_chunks(
        pages_data=pages_data,
        filename=safe_name,
        document_id=doc_id,
        session_id=sess_id,
        user_id=clean_user_id,
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        embedding_model=embedding_service.model_name,
        embedding_dim=embedding_service.expected_dim
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks created from document.")

    texts_to_embed = [c["text"] for c in chunks]
    embeddings = embedding_service.generate_embeddings(texts_to_embed)

    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    return {
        "status": "processed",
        "document_id": doc_id,
        "filename": safe_name,
        "user_id": clean_user_id,
        "chunks_count": len(chunks),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "extraction_report": parse_result["extraction_report"]
    }

@app.post("/api/chat")
def chat_query(req: ChatRequest):
    clean_user_id = req.user_id.strip().lower() if req.user_id else "anonymous_user"
    session_id = req.session_id or f"sess_{int(time.time())}"
    safe_name = sanitize_filename(req.filename) if req.filename else None

    recent_messages = history_store.get_messages(session_id=session_id)

    history_store.create_session(
        session_id=session_id,
        user_id=clean_user_id,
        document_id=req.document_id or "",
        filename=safe_name or "General Document",
        system_prompt=req.system_prompt or ""
    )
    history_store.add_message(session_id=session_id, role="user", content=req.query, document_id=req.document_id or "")

    rag_result = rag_engine.answer_query(
        query=req.query,
        filename=safe_name,
        document_id=req.document_id,
        session_id=session_id,
        user_id=clean_user_id,
        system_prompt=req.system_prompt,
        top_k=req.top_k,
        temperature=req.temperature,
        chat_history=recent_messages
    )

    history_store.add_message(
        session_id=session_id,
        role="assistant",
        content=rag_result["answer"],
        document_id=req.document_id or "",
        sources=rag_result["sources"],
        evidence_quotes=rag_result.get("verified_quotes", []),
        is_untrusted_assistant=True
    )

    res = {
        "session_id": session_id,
        "user_id": clean_user_id,
        "document_id": req.document_id,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "verified_quotes": rag_result.get("verified_quotes", []),
        "confidence": rag_result.get("confidence", 0.0),
        "retrieved_count": rag_result["retrieved_count"]
    }

    if DEBUG_RAG and "rag_trace" in rag_result:
        res["rag_trace"] = rag_result["rag_trace"]

    return res

@app.get("/api/bucket/files")
def list_bucket_files(user_id: str):
    clean_user_id = user_id.strip().lower() if user_id else ""
    files = storage_bucket.list_files(user_id=clean_user_id)
    return {
        "bucket_name": storage_bucket.bucket_name,
        "user_id": clean_user_id,
        "files_count": len(files),
        "files": files
    }

@app.delete("/api/bucket/files/{filename}")
def delete_bucket_file(filename: str):
    safe_name = sanitize_filename(filename)
    deleted = storage_bucket.delete_file(safe_name)
    vector_store.clear_document(filename=safe_name)
    return {"status": "deleted" if deleted else "not_found", "filename": safe_name}

@app.get("/api/sessions")
def list_user_sessions(user_id: str):
    clean_user_id = user_id.strip().lower() if user_id else ""
    sessions = history_store.list_sessions(user_id=clean_user_id)
    return sessions

@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = history_store.get_messages(session_id)
    return {"session": session, "messages": messages}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = history_store.delete_session(session_id)
    return {"status": "deleted" if deleted else "not_found", "session_id": session_id}
