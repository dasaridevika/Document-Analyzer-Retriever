from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
import logging
import gc

from backend.services.pdf_parser import PDFParser
from backend.services.chunker import DocumentChunker
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStoreManager
from backend.services.history_store import HistoryStore
from backend.services.storage_bucket import StorageBucketManager
from backend.services.rag_engine import RAGEngine
from backend.services.worker_analyzer import analyze_extracted_data
from backend.config import DEFAULT_SYSTEM_PROMPT, BACKEND_HOST, BACKEND_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc_analyser_backend")

app = FastAPI(
    title="Document Analyser & Retriever API",
    description="S3-Exclusive Storage Bucket RAG Backend with PyMuPDF & Cloudflare Workers AI",
    version="2.0.0"
)

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
embedding_service = EmbeddingService()
vector_store = VectorStoreManager()
history_store = HistoryStore()
storage_bucket = StorageBucketManager()
rag_engine = RAGEngine(embedding_service=embedding_service, vector_store=vector_store)

# In-memory document metadata cache
document_cache: Dict[str, Dict[str, Any]] = {}

# Pydantic Schemas
class ProcessRequest(BaseModel):
    filename: str
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    filename: Optional[str] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    top_k: int = 8
    temperature: float = 0.2
    chunk_strategy: Optional[str] = "recursive"
    chunk_size: Optional[int] = 500

class AnalyzeRequest(BaseModel):
    filename: str
    url: Optional[str] = ""
    title: Optional[str] = ""
    analysis_type: Optional[str] = "summary"

# Endpoints

@app.get("/api/health")
def health_check():
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "vector_store_stats": stats,
        "embedding_model": embedding_service.model_name,
        "bucket_name": storage_bucket.bucket_name
    }

def _save_and_parse_pdf_to_s3(file: UploadFile) -> Dict[str, Any]:
    """
    Saves PDF file directly to S3 Storage Bucket and parses text in memory.
    """
    file.file.seek(0)
    file_bytes = file.file.read()

    # 1. Save directly into S3 Storage Bucket
    bucket_info = storage_bucket.save_file(file.filename, file_bytes)

    # 2. Parse PDF text from bytes directly
    parsed_doc = PDFParser.parse_pdf_bytes(file_bytes, file.filename)
    parsed_doc["bucket_info"] = bucket_info
    gc.collect()
    return parsed_doc

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads PDF file directly into the S3 Storage Bucket.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        parsed_doc = await run_in_threadpool(_save_and_parse_pdf_to_s3, file)
        document_cache[file.filename] = parsed_doc

        return {
            "message": f"PDF '{file.filename}' uploaded directly to S3 Storage Bucket successfully.",
            "filename": file.filename,
            "bucket_info": parsed_doc.get("bucket_info"),
            "metadata": parsed_doc["metadata"],
            "page_count": len(parsed_doc["pages"])
        }
    except Exception as e:
        logger.error(f"S3 Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"S3 Storage Bucket upload error: {str(e)}")

@app.post("/api/process")
async def process_chunks(req: ProcessRequest):
    filename = req.filename
    if filename not in document_cache:
        file_bytes = await run_in_threadpool(storage_bucket.get_file, filename)
        if file_bytes:
            parsed_doc = await run_in_threadpool(PDFParser.parse_pdf_bytes, file_bytes, filename)
            document_cache[filename] = parsed_doc
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in S3 Storage Bucket.")

    doc_data = document_cache[filename]
    pages_data = doc_data["pages"]

    # 1. Chunking
    chunks = await run_in_threadpool(
        DocumentChunker.chunk_document,
        pages_data, filename, req.strategy, req.chunk_size, req.chunk_overlap
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks could be generated from document.")

    # 2. Clear old vectors for document if re-processing
    await run_in_threadpool(vector_store.clear_document, filename)

    # 3. Generate Embeddings & Index in threadpool
    chunk_texts = [c["text"] for c in chunks]
    embeddings = await run_in_threadpool(embedding_service.generate_embeddings, chunk_texts)
    await run_in_threadpool(vector_store.add_chunks, chunks, embeddings)

    doc_data["chunks"] = chunks
    gc.collect()

    return {
        "message": f"Successfully chunked and indexed {len(chunks)} text chunks for '{filename}' in S3 Storage Bucket.",
        "filename": filename,
        "strategy": req.strategy,
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0] if chunks else None
    }

@app.post("/api/analyze")
async def analyze_document_endpoint(req: AnalyzeRequest):
    filename = req.filename
    if filename not in document_cache:
        file_bytes = await run_in_threadpool(storage_bucket.get_file, filename)
        if file_bytes:
            parsed_doc = await run_in_threadpool(PDFParser.parse_pdf_bytes, file_bytes, filename)
            document_cache[filename] = parsed_doc
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in S3 Storage Bucket.")

    doc_text = document_cache[filename].get("full_text", "")
    analysis_result = await analyze_extracted_data(
        url=req.url or filename,
        title=req.title or filename,
        extracted_text=doc_text,
        analysis_type=req.analysis_type
    )

    document_cache[filename]["analysis"] = analysis_result
    return {
        "filename": filename,
        "analysis": analysis_result
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    filename = req.filename or "General Document"

    if not session_id:
        session_id = await run_in_threadpool(
            history_store.create_session,
            filename, req.system_prompt, req.chunk_strategy, req.chunk_size
        )

    # Save User Message to History
    await run_in_threadpool(history_store.add_message, session_id, "user", req.query)

    # RAG Retrieval & Answer Synthesis
    rag_result = await run_in_threadpool(
        rag_engine.answer_query,
        req.query, req.filename, req.system_prompt, req.top_k, req.temperature
    )

    # Save Assistant Response & Sources to History
    await run_in_threadpool(
        history_store.add_message,
        session_id, "assistant", rag_result["answer"], rag_result["sources"]
    )

    return {
        "session_id": session_id,
        "query": req.query,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "system_prompt_used": rag_result["system_prompt_used"],
        "retrieved_count": rag_result["retrieved_count"]
    }

@app.get("/api/bucket/files")
async def list_bucket_files():
    files = await run_in_threadpool(storage_bucket.list_files)
    return {
        "bucket_name": storage_bucket.bucket_name,
        "files": files
    }

@app.delete("/api/bucket/files/{filename}")
async def delete_bucket_file(filename: str):
    deleted = await run_in_threadpool(storage_bucket.delete_file, filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found in S3 Storage Bucket.")
    return {"message": f"Deleted '{filename}' from S3 Storage Bucket."}

@app.get("/api/sessions")
async def list_sessions():
    return await run_in_threadpool(history_store.list_sessions)

@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    session = await run_in_threadpool(history_store.get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = await run_in_threadpool(history_store.get_messages, session_id)
    return {
        "session": session,
        "messages": messages
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = await run_in_threadpool(history_store.delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted successfully."}

@app.get("/api/chunks/{filename}")
def get_document_chunks(filename: str):
    if filename not in document_cache or "chunks" not in document_cache[filename]:
        raise HTTPException(status_code=404, detail="Chunks not available for this document.")
    return {
        "filename": filename,
        "chunks": document_cache[filename]["chunks"]
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
