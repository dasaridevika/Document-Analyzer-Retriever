import streamlit as st
import requests
import os
import json

# Page Config
st.set_page_config(
    page_title="DocAnalyser AI | RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State Initialization
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You are a precise, highly analytical Document AI assistant.
Your goal is to answer the user's questions strictly based on the provided document context.
If the answer cannot be found in the context, explicitly state that the document does not contain that information.
Always cite section titles or page numbers when quoting facts."""
if "chunks_data" not in st.session_state:
    st.session_state.chunks_data = []

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/document-file.png", width=64)
    st.title("DocAnalyser AI")
    st.caption("Cloudflare Workers AI, RAG & Storage Bucket Engine")
    st.divider()

    # System Prompt Presets
    st.subheader("⚙️ System Prompt Customizer")
    preset = st.selectbox(
        "Choose Preset:",
        ["Custom", "Academic Analyst", "Executive Summary", "Concise Extractor", "Deep RAG Explainer"]
    )

    if preset == "Academic Analyst":
        st.session_state.system_prompt = "You are an academic researcher. Analyze the document context critically, evaluate evidence, cite specific pages, and highlight any potential limitations in the data."
    elif preset == "Executive Summary":
        st.session_state.system_prompt = "You are a business executive advisor. Provide high-level bulleted summaries based on the document context, emphasizing actionable takeaways and key metrics."
    elif preset == "Concise Extractor":
        st.session_state.system_prompt = "You are a precise data extractor. Answer user questions in under 3 concise sentences, focusing strictly on facts directly mentioned in the document context."
    elif preset == "Deep RAG Explainer":
        st.session_state.system_prompt = "You are an educational tutor. Explain concepts thoroughly using context from the document, breaking down complex topics into clear step-by-step explanations."

    custom_prompt = st.text_area(
        "Active System Prompt:",
        value=st.session_state.system_prompt,
        height=140,
        help="Update the system prompt live to guide how the RAG model interprets document context!"
    )
    st.session_state.system_prompt = custom_prompt

    st.divider()

    # Chunking & RAG Hyperparameters
    st.subheader("🧩 Chunking & Retrieval Setup")
    chunk_strategy = st.selectbox(
        "Chunking Strategy:",
        ["recursive", "fixed", "page_aware", "semantic"],
        help="Recursive: Hierarchical splits\nFixed: Sliding window with overlap\nPage-Aware: Scoped to PDF pages\nSemantic: Logical paragraph grouping"
    )
    chunk_size = st.slider("Chunk Size (Chars):", 200, 1500, 500, step=50)
    chunk_overlap = st.slider("Overlap (Chars):", 0, 300, 50, step=10)
    top_k = st.slider("Top-K Retrieval:", 1, 10, 4)
    temperature = st.slider("LLM Temperature:", 0.0, 1.0, 0.2, step=0.05)

    st.divider()

    # Saved Conversations Manager (Railway Volume persistent storage)
    st.subheader("💾 Saved Sessions (Railway Storage)")
    try:
        resp = requests.get(f"{BACKEND_URL}/api/sessions", timeout=5)
        if resp.status_code == 200:
            saved_sessions = resp.json()
            if saved_sessions:
                session_options = {
                    f"{s['filename']} ({s['session_id']}) - {s['message_count']} msgs": s['session_id']
                    for s in saved_sessions
                }
                selected_sess_label = st.selectbox("Load Saved Session:", ["None"] + list(session_options.keys()))
                if selected_sess_label != "None":
                    target_id = session_options[selected_sess_label]
                    if st.button("🔄 Load Selected Session", use_container_width=True):
                        sess_resp = requests.get(f"{BACKEND_URL}/api/sessions/{target_id}")
                        if sess_resp.status_code == 200:
                            sess_data = sess_resp.json()
                            st.session_state.session_id = target_id
                            st.session_state.current_filename = sess_data["session"]["filename"]
                            st.session_state.system_prompt = sess_data["session"]["system_prompt"]
                            st.session_state.messages = [
                                {
                                    "role": m["role"],
                                    "content": m["content"],
                                    "sources": m.get("sources", [])
                                }
                                for m in sess_data["messages"]
                            ]
                            st.success(f"Session {target_id} loaded!")
                            st.rerun()
            else:
                st.caption("No saved sessions found in storage.")
    except Exception:
        st.warning("Backend offline or starting up...")

    if st.button("➕ Start New Session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.analysis_data = None
        st.success("New session initialized.")
        st.rerun()

# --- MAIN DASHBOARD HEADER ---
st.markdown("""
<div class="main-header">
    <h1>📄 Document Analyser & RAG Retriever</h1>
    <p>Powered by PyMuPDF PDF Chunking, Cloudflare Workers AI Embeddings (bge-large-en-v1.5), Cloudflare R2 / S3 Storage Bucket, and Persistent Railway Volume Storage.</p>
</div>
""", unsafe_allow_html=True)

# TABS NAVIGATION
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 1. Upload & Analyze PDF",
    "💬 2. RAG Chatbot",
    "📦 3. Storage Bucket Manager",
    "📚 4. Saved History Browser",
    "📊 5. Free-Tier Token Monitor"
])

# --- TAB 1: UPLOAD & CHUNK PDF ---
with tab1:
    st.subheader("Upload PDF Document to Storage Bucket")
    uploaded_file = st.file_uploader("Choose a PDF file to analyze:", type=["pdf"])

    col1, col2 = st.columns([1, 1])

    with col1:
        if uploaded_file is not None:
            if st.button("🚀 Upload to Bucket & Index", type="primary", use_container_width=True):
                with st.spinner("Saving PDF to Storage Bucket & extracting text..."):
                    # Upload PDF
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    up_resp = requests.post(f"{BACKEND_URL}/api/upload", files=files)

                    if up_resp.status_code == 200:
                        up_data = up_resp.json()
                        st.session_state.current_filename = up_data["filename"]
                        st.session_state.doc_metadata = up_data["metadata"]

                        # Apply Chunking & Vector Indexing
                        with st.spinner(f"Chunking via '{chunk_strategy}' strategy & embedding with Cloudflare Workers AI (bge-large-en-v1.5)..."):
                            proc_payload = {
                                "filename": up_data["filename"],
                                "strategy": chunk_strategy,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap
                            }
                            proc_resp = requests.post(f"{BACKEND_URL}/api/process", json=proc_payload)
                            if proc_resp.status_code == 200:
                                proc_data = proc_resp.json()
                                st.session_state.chunks_data = proc_data.get("sample_chunk", [])
                                storage_type = up_data.get("bucket_info", {}).get("storage_type", "Storage Bucket")
                                st.success(f"Successfully stored in '{storage_type}' and processed {proc_data['chunk_count']} chunks for '{up_data['filename']}'!")
                            else:
                                st.error(f"Processing error: {proc_resp.text}")
                    else:
                        st.error(f"Upload error: {up_resp.text}")

    with col2:
        if st.session_state.doc_metadata:
            meta = st.session_state.doc_metadata
            st.markdown("### Document Stats")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><h3>Total Pages</h3><p>{meta.get('total_pages', 0)}</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><h3>Word Count</h3><p>{meta.get('total_words', 0):,}</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><h3>Total Chars</h3><p>{meta.get('total_chars', 0):,}</p></div>", unsafe_allow_html=True)

    # Cloudflare AI Deep Analysis Section
    if st.session_state.current_filename:
        st.divider()
        st.subheader("✨ Cloudflare AI Deep Document Analysis")
        if st.button("🔍 Run Full AI Document Analysis (Summary, Keywords, Sentiment, Actions)", use_container_width=True):
            with st.spinner("Analyzing document content via Cloudflare Workers AI..."):
                an_resp = requests.post(f"{BACKEND_URL}/api/analyze", json={"filename": st.session_state.current_filename})
                if an_resp.status_code == 200:
                    st.session_state.analysis_data = an_resp.json().get("analysis", {})
                    st.success("AI Analysis Complete!")
                else:
                    st.error(f"Analysis failed: {an_resp.text}")

        if st.session_state.analysis_data:
            an = st.session_state.analysis_data
            ac1, ac2 = st.columns([2, 1])
            with ac1:
                st.markdown(f"**Executive Summary:**\n> {an.get('summary', 'N/A')}")
                if an.get("important_points"):
                    st.markdown("**Key Takeaways:**")
                    for p in an.get("important_points", []):
                        st.markdown(f"- {p}")
                if an.get("action_items"):
                    st.markdown("**Action Items:**")
                    for a in an.get("action_items", []):
                        st.markdown(f"-[ ] {a}")
            with ac2:
                st.markdown(f"**Sentiment:** `{an.get('sentiment', 'neutral').upper()}`")
                if an.get("topics"):
                    st.markdown(f"**Topics:** {', '.join(an.get('topics', []))}")
                if an.get("keywords"):
                    st.markdown(f"**Keywords:** {', '.join(an.get('keywords', []))}")

        st.divider()
        st.subheader("🔍 Chunk Strategy Visualizer")
        if st.button("Inspect Generated Chunks"):
            c_resp = requests.get(f"{BACKEND_URL}/api/chunks/{st.session_state.current_filename}")
            if c_resp.status_code == 200:
                chunks = c_resp.json().get("chunks", [])
                st.caption(f"Showing {len(chunks)} parsed chunks:")
                for i, c in enumerate(chunks[:10]):
                    with st.expander(f"Chunk #{c['chunk_index']} | Page {c['page_number']} | Length: {c['char_count']} chars"):
                        st.code(c["text"], language="text")

# --- TAB 2: RAG CHATBOT ---
with tab2:
    st.markdown(f"""
    <div class="system-prompt-badge">
        🎯 <b>Active System Prompt:</b> {st.session_state.system_prompt[:140]}...
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_filename:
        st.info(f"📁 Active Document: **{st.session_state.current_filename}**")
    else:
        st.warning("⚠️ No document uploaded yet. You can still ask general questions or select a PDF from the Storage Bucket.")

    # Display Message History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Retrieved Sources & Citations"):
                    for src in msg["sources"]:
                        score = src.get("similarity_score", 0.0)
                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="citation-header">Source #{src.get('source_id')} | Page {src.get('page_number')} | Similarity: {score:.2f}</div>
                            {src.get('text')}
                        </div>
                        """, unsafe_allow_html=True)

    # User Input Chat Box
    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query Backend RAG Engine
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant document chunks & generating answer..."):
                chat_payload = {
                    "session_id": st.session_state.session_id,
                    "query": user_query,
                    "filename": st.session_state.current_filename,
                    "system_prompt": st.session_state.system_prompt,
                    "top_k": top_k,
                    "temperature": temperature,
                    "chunk_strategy": chunk_strategy,
                    "chunk_size": chunk_size
                }

                try:
                    resp = requests.post(f"{BACKEND_URL}/api/chat", json=chat_payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.session_id = data["session_id"]
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 View Retrieved Sources & Citations"):
                                for src in sources:
                                    score = src.get("similarity_score", 0.0)
                                    st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-header">Source #{src.get('source_id')} | Page {src.get('page_number')} | Similarity: {score:.2f}</div>
                                        {src.get('text')}
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"Chat API error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")

# --- TAB 3: STORAGE BUCKET MANAGER ---
with tab3:
    st.subheader("📦 Cloudflare R2 / Storage Bucket File Browser")
    try:
        b_resp = requests.get(f"{BACKEND_URL}/api/bucket/files")
        if b_resp.status_code == 200:
            b_data = b_resp.json()
            bucket_name = b_data.get("bucket_name", "Storage Bucket")
            files = b_data.get("files", [])

            st.caption(f"Active Storage Bucket: `{bucket_name}` | Total Files: `{len(files)}`")

            if files:
                for f in files:
                    with st.expander(f"📄 {f['filename']} ({f['size_bytes']:,} bytes) - {f.get('storage_type', 'Storage')}"):
                        st.markdown(f"**Last Modified:** `{f.get('last_modified', 'N/A')}`")
                        st.markdown(f"**Storage Engine:** `{f.get('storage_type', 'Storage')}`")

                        col_x, col_y = st.columns([1, 1])
                        with col_x:
                            if st.button("🚀 Load & Process this PDF", key=f"proc_bucket_{f['filename']}"):
                                with st.spinner(f"Re-indexing '{f['filename']}' from storage bucket..."):
                                    p_resp = requests.post(f"{BACKEND_URL}/api/process", json={
                                        "filename": f['filename'],
                                        "strategy": chunk_strategy,
                                        "chunk_size": chunk_size,
                                        "chunk_overlap": chunk_overlap
                                    })
                                    if p_resp.status_code == 200:
                                        st.session_state.current_filename = f['filename']
                                        st.success(f"'{f['filename']}' loaded from storage bucket and indexed!")
                                        st.rerun()
                                    else:
                                        st.error(f"Error loading from bucket: {p_resp.text}")
                        with col_y:
                            if st.button("🗑️ Delete from Bucket", key=f"del_bucket_{f['filename']}"):
                                requests.delete(f"{BACKEND_URL}/api/bucket/files/{f['filename']}")
                                st.success(f"Deleted '{f['filename']}' from bucket.")
                                st.rerun()
            else:
                st.info("No files currently stored in the storage bucket. Upload a PDF in Tab 1.")
    except Exception as e:
        st.error(f"Failed to connect to storage bucket API: {e}")

# --- TAB 4: SAVED HISTORY BROWSER ---
with tab4:
    st.subheader("📚 Saved Conversations in Railway Storage")
    try:
        resp = requests.get(f"{BACKEND_URL}/api/sessions")
        if resp.status_code == 200:
            sessions = resp.json()
            if sessions:
                for s in sessions:
                    with st.expander(f"💬 Session `{s['session_id']}` | Document: {s['filename']} | {s['message_count']} messages"):
                        st.markdown(f"**Created At:** `{s['created_at']}`")
                        st.markdown(f"**System Prompt:** *{s['system_prompt']}*")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            if st.button("📥 Load into Chat", key=f"load_{s['session_id']}"):
                                sess_resp = requests.get(f"{BACKEND_URL}/api/sessions/{s['session_id']}")
                                if sess_resp.status_code == 200:
                                    sess_data = sess_resp.json()
                                    st.session_state.session_id = s['session_id']
                                    st.session_state.current_filename = s['filename']
                                    st.session_state.system_prompt = s['system_prompt']
                                    st.session_state.messages = [
                                        {
                                            "role": m["role"],
                                            "content": m["content"],
                                            "sources": m.get("sources", [])
                                        }
                                        for m in sess_data["messages"]
                                    ]
                                    st.success("Session loaded successfully! Switch to Tab 2.")
                        with col_b:
                            if st.button("🗑️ Delete Session", key=f"del_{s['session_id']}"):
                                requests.delete(f"{BACKEND_URL}/api/sessions/{s['session_id']}")
                                st.success("Session deleted.")
                                st.rerun()
            else:
                st.info("No saved chat history found in Railway storage bucket database.")
    except Exception as e:
        st.error(f"Failed to fetch session history: {e}")

# --- TAB 5: TOKEN MONITOR ---
with tab5:
    st.subheader("📊 Free Tier Token & Neuron Monitor")
    st.markdown("""
    ### Cloudflare Workers AI Free Tier Limits:
    - **Neurons Daily Allowance:** 10,000 Neurons / Day (Free Tier)
    - **Embedding Model:** `@cf/baai/bge-large-en-v1.5` (~1,024 vector dimensions)
    - **LLM Model:** `@cf/meta/llama-3-8b-instruct`
    - **Storage Bucket:** Cloudflare R2 Free Tier (10 GB storage free, 0 egress fees)
    """)

    st.divider()
    if st.session_state.doc_metadata:
        words = st.session_state.doc_metadata.get("total_words", 0)
        est_tokens = int(words * 1.3)
        st.markdown(f"**Current Document Estimated Tokens:** `{est_tokens:,} tokens`")
        st.markdown(f"**Estimated Embedding Neurons Used:** `~{int(est_tokens / 100)} Neurons`")
        st.progress(min(1.0, est_tokens / 100000))
