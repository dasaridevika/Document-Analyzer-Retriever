from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging

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
    description="RAG Backend with Cloudflare R2 / S3 Storage Bucket, PyMuPDF, Cloudflare Workers AI Embeddings (@cf/baai/bge-large-en-v1.5), Vector Storage, & Chat Persistence",
    version="1.2.0"
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

# In-memory document buffer cache for active session processing
document_cache: Dict[str, Dict[str, Any]] = {}

# Pydantic Schemas
class ProcessRequest(BaseModel):
    filename: str
    strategy: str = "recursive"  # fixed, recursive, page_aware, semantic
    chunk_size: int = 500
    chunk_overlap: int = 50

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    filename: Optional[str] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    top_k: int = 4
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

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Parses an uploaded PDF file, saves it into Cloudflare R2 / Storage Bucket, and caches text metadata.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        
        # 1. Save file to Storage Bucket (Cloudflare R2 / S3 / Local Volume)
        bucket_info = storage_bucket.save_file(file.filename, contents)

        # 2. Parse PDF via PyMuPDF
        parsed_doc = PDFParser.parse_pdf_bytes(contents, file.filename)
        parsed_doc["bucket_info"] = bucket_info
        
        # Cache parsed doc in memory
        document_cache[file.filename] = parsed_doc

        return {
            "message": "PDF uploaded, stored in bucket, and parsed successfully",
            "filename": file.filename,
            "bucket_info": bucket_info,
            "metadata": parsed_doc["metadata"],
            "page_count": len(parsed_doc["pages"])
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/process")
def process_chunks(req: ProcessRequest):
    """
    Applies selected chunking strategy to PDF text and indexes vectors into ChromaDB using BGE Large embeddings.
    """
    filename = req.filename
    if filename not in document_cache:
        # Try fetching file bytes from Storage Bucket
        file_bytes = storage_bucket.get_file(filename)
        if file_bytes:
            parsed_doc = PDFParser.parse_pdf_bytes(file_bytes, filename)
            document_cache[filename] = parsed_doc
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in bucket or memory.")

    doc_data = document_cache[filename]
    pages_data = doc_data["pages"]

    # 1. Chunking
    chunks = DocumentChunker.chunk_document(
        pages_data=pages_data,
        filename=filename,
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks could be generated from document.")

    # 2. Clear old vectors for document if re-processing
    vector_store.clear_document(filename)

    # 3. Generate Embeddings & Index
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedding_service.generate_embeddings(chunk_texts)
    vector_store.add_chunks(chunks, embeddings)

    # Store generated chunks inside cached doc data
    doc_data["chunks"] = chunks

    return {
        "message": f"Successfully chunked and indexed {len(chunks)} text chunks.",
        "filename": filename,
        "strategy": req.strategy,
        "chunk_count": len(chunks),
        "sample_chunk": chunks[0] if chunks else None
    }

@app.post("/api/analyze")
async def analyze_document_endpoint(req: AnalyzeRequest):
    """
    Runs Cloudflare Worker LLM analysis (summary, topics, keywords, sentiment, key takeaways, action items) on document text.
    """
    filename = req.filename
    if filename not in document_cache:
        file_bytes = storage_bucket.get_file(filename)
        if file_bytes:
            parsed_doc = PDFParser.parse_pdf_bytes(file_bytes, filename)
            document_cache[filename] = parsed_doc
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

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
def chat_endpoint(req: ChatRequest):
    """
    RAG Chat endpoint with dynamic system prompt, retrieval, and persistent chat history saving.
    """
    session_id = req.session_id
    filename = req.filename or "General Document"

    if not session_id:
        session_id = history_store.create_session(
            filename=filename,
            system_prompt=req.system_prompt,
            chunk_strategy=req.chunk_strategy,
            chunk_size=req.chunk_size
        )

    # Save User Message to History
    history_store.add_message(session_id, role="user", content=req.query)

    # RAG Retrieval & Answer Synthesis
    rag_result = rag_engine.answer_query(
        query=req.query,
        filename=req.filename,
        system_prompt=req.system_prompt,
        top_k=req.top_k,
        temperature=req.temperature
    )

    # Save Assistant Response & Sources to History
    history_store.add_message(
        session_id=session_id,
        role="assistant",
        content=rag_result["answer"],
        sources=rag_result["sources"]
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
def list_bucket_files():
    """
    Lists all PDF files stored in Cloudflare R2 / S3 Storage Bucket.
    """
    return {
        "bucket_name": storage_bucket.bucket_name,
        "files": storage_bucket.list_files()
    }

@app.delete("/api/bucket/files/{filename}")
def delete_bucket_file(filename: str):
    """
    Deletes a PDF file from the Storage Bucket.
    """
    deleted = storage_bucket.delete_file(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found in storage bucket.")
    return {"message": f"Deleted '{filename}' from storage bucket."}

@app.get("/api/sessions")
def list_sessions():
    return history_store.list_sessions()

@app.get("/api/sessions/{session_id}")
def get_session_details(session_id: str):
    session = history_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = history_store.get_messages(session_id)
    return {
        "session": session,
        "messages": messages
    }

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = history_store.delete_session(session_id)
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
