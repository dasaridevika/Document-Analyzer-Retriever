import os
import time
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend.config import DATA_DIR, UPLOAD_DIR
from backend.services.pdf_loader import PDFLoader
from backend.services.chunker import TextChunker
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStoreManager
from backend.services.rag_engine import RAGEngine
from backend.services.history_store import HistoryStore
from backend.services.storage_bucket import StorageBucketManager
from backend.services.auth_firebase import FirebaseAuthService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DocAnalyzerBackend")

app = FastAPI(title="DocAnalyzer RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services Initialization
pdf_loader = PDFLoader()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(embedding_service=embedding_service, vector_store=vector_store)
history_store = HistoryStore()
storage_bucket = StorageBucketManager()
auth_service = FirebaseAuthService()

# Request Models
class ProcessRequest(BaseModel):
    filename: str
    user_id: str = "anonymous_user"
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str = "anonymous_user"
    query: str
    filename: Optional[str] = None
    system_prompt: Optional[str] = None
    top_k: int = 8
    temperature: float = 0.1

class VerifyAuthRequest(BaseModel):
    token_or_email: str

@app.get("/api/health")
def health_check():
    stats = vector_store.get_stats()
    return {
        "status": "online",
        "service": "DocAnalyzer API",
        "vector_stats": stats,
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

    # Save user profile permanently in Storage Bucket
    profile = storage_bucket.save_user_profile(user_info["email"], user_info.get("name", ""))

    return {"status": "authenticated", "user": user_info, "profile": profile}

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous_user")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    clean_user_id = user_id.strip().lower() if user_id else "anonymous_user"

    # Save directly to Railway Storage Bucket
    bucket_result = storage_bucket.save_file(
        filename=file.filename,
        content=content,
        user_id=clean_user_id,
        content_type="application/pdf"
    )

    # Extract metadata preview
    meta = pdf_loader.extract_metadata(content, file.filename)

    return {
        "filename": file.filename,
        "user_id": clean_user_id,
        "size_bytes": len(content),
        "bucket_info": bucket_result,
        "metadata": meta
    }

@app.post("/api/process")
def process_document(req: ProcessRequest):
    clean_user_id = req.user_id.strip().lower() if req.user_id else "anonymous_user"
    pdf_bytes = storage_bucket.get_file(req.filename)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail=f"File '{req.filename}' not found in Storage Bucket.")

    pages_data = pdf_loader.extract_pages(pdf_bytes, req.filename)
    if not pages_data:
        raise HTTPException(status_code=400, detail="Failed to extract text pages from PDF.")

    chunks = chunker.create_chunks(
        pages_data=pages_data,
        filename=req.filename,
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks created from document.")

    texts_to_embed = [c["text"] for c in chunks]
    embeddings = embedding_service.generate_embeddings(texts_to_embed)

    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    return {
        "status": "processed",
        "filename": req.filename,
        "user_id": clean_user_id,
        "chunks_count": len(chunks),
        "embedding_dim": len(embeddings[0]) if embeddings else 0
    }

@app.post("/api/chat")
def chat_query(req: ChatRequest):
    clean_user_id = req.user_id.strip().lower() if req.user_id else "anonymous_user"
    session_id = req.session_id or f"sess_{int(time.time())}"

    # Record User Message
    history_store.create_session(
        session_id=session_id,
        user_id=clean_user_id,
        filename=req.filename or "General Document",
        system_prompt=req.system_prompt or ""
    )
    history_store.add_message(session_id=session_id, role="user", content=req.query)

    # RAG Generation
    rag_result = rag_engine.answer_query(
        query=req.query,
        filename=req.filename,
        system_prompt=req.system_prompt,
        top_k=req.top_k,
        temperature=req.temperature
    )

    # Record Assistant Message
    history_store.add_message(
        session_id=session_id,
        role="assistant",
        content=rag_result["answer"],
        sources=rag_result["sources"]
    )

    return {
        "session_id": session_id,
        "user_id": clean_user_id,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "retrieved_count": rag_result["retrieved_count"]
    }

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
    deleted = storage_bucket.delete_file(filename)
    vector_store.clear_document(filename)
    return {"status": "deleted" if deleted else "not_found", "filename": filename}

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
