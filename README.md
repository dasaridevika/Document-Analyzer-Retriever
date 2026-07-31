# 📄 Document Analyser & RAG Retriever

An end-to-end, free-tier conscious **Document Analyser and Retriever Chatbot** featuring **FastAPI**, **Streamlit**, **PyMuPDF**, **Cloudflare Workers AI** embedding models (`bge-large-en-v1.5` / `embeddinggemma-300m`), vector storage, persistent chat session history on **Railway Volume Storage**, and dynamic system prompt customization.

---

## 🌟 Key Features

1. **PyMuPDF (`fitz`) PDF Ingestion**: Page-by-page text extraction, word/character analytics, and structural clean-up.
2. **Multi-Strategy RAG Chunking**:
   - **Fixed Size Window**: Sliding window with configurable character overlap.
   - **Recursive Character**: Hierarchical split by paragraphs, sentences, and words.
   - **Page-Aware**: Keeps text chunks bound strictly to original PDF pages.
   - **Semantic Paragraph**: Groups logical paragraphs within specified token thresholds.
3. **Cloudflare Workers AI Integration**:
   - Primary Embeddings: `@cf/baai/bge-large-en-v1.5` or `@cf/google/embeddinggemma-300m` via REST API.
   - Fallback: Local `sentence-transformers` (`all-MiniLM-L6-v2`) if credentials are absent.
4. **Persistent Conversation Storage on Railway Volume**:
   - SQLite history storage (`/data/chat_history.db`) saves all chat sessions, questions, retrieved citations, and AI answers across restarts.
   - Streamlit frontend **Saved History Browser** to search, reload, and switch between past document analysis sessions anytime.
5. **Dynamic System Prompt Editor**: Real-time system prompt modification directly from the UI sidebar with presets (Academic Analyst, Executive Summary, Concise Extractor, Deep RAG Explainer).
6. **Free-Tier Conscious**: Batching requests, token budget estimators, and zero external DB cost by utilizing persistent disk volumes.

---

## 🏗️ Repository Structure

```
doc-analyser-rag/
├── backend/
│   ├── config.py           # Configuration & environment settings
│   ├── main.py             # FastAPI REST endpoints & CORS middleware
│   └── services/
│       ├── pdf_parser.py   # PyMuPDF text & page parser
│       ├── chunker.py      # RAG Document Chunking algorithms
│       ├── embeddings.py   # Cloudflare Workers AI embedding client
│       ├── vector_store.py # Persistent ChromaDB vector DB wrapper
│       ├── history_store.py# SQLite Chat History persistence store
│       └── rag_engine.py   # Context retrieval & LLM synthesis engine
├── frontend/
│   ├── app.py              # Streamlit UI app
│   └── style.css           # Glassmorphic dark theme stylesheet
├── Dockerfile              # Container definition for Railway deployment
├── railway.toml            # Railway config file
├── Procfile                # Heroku/PaaS start command
├── requirements.txt        # Python dependencies
└── README.md               # Documentation & Railway guide
```

---

## 🚀 Local Development Guide

### 1. Clone & Setup Virtual Environment

```bash
git clone <your-github-repo-url>
cd doc-analyser-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Add your Cloudflare Workers AI credentials (optional - local fallback will run automatically if omitted):
```env
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-large-en-v1.5
CLOUDFLARE_LLM_MODEL=@cf/meta/llama-3-8b-instruct
```

### 3. Run FastAPI Backend & Streamlit Frontend

Open two terminal windows:

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
streamlit run frontend/app.py
```

Access the web app at `http://localhost:8501`.

---

## ☁️ Deployment Guide for Railway App

### Step 1: Push Project to GitHub

```bash
git init
git add .
git commit -m "Initial commit of Document Analyser & RAG Retriever"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### Step 2: Deploy on Railway

1. Log into your [Railway Dashboard](https://railway.app/).
2. Click **+ New Project** -> **Deploy from GitHub repo**.
3. Select your repository `doc-analyser-rag`.
4. Railway will automatically detect the `Dockerfile` and `railway.toml`.

### Step 3: Attach Railway Volume (Persistent Storage for History & Vectors)

1. In your Railway project service, click **+ New** -> **Volume**.
2. Mount the volume path to `/app/storage` (or `/data`).
3. Set environment variable:
   ```env
   DATA_DIR=/app/storage
   ```

### Step 4: Add Cloudflare Workers AI Credentials in Railway

In your Railway service **Variables** tab, add:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_EMBEDDING_MODEL` = `@cf/baai/bge-large-en-v1.5`

Railway will build and deploy your app. You will get a live public domain URL!

---

## 💡 Free Tier Token & Resource Optimization

- **Cloudflare Workers AI**: Offers 10,000 free neurons/day.
- **Batch Embedding Requests**: The embedding service sends up to 32 chunks per REST payload to maximize neuron efficiency.
- **Persistent Storage**: All chat history and vector embeddings reside on Railway Volume disk storage (`/app/storage`), eliminating paid external database fees.
