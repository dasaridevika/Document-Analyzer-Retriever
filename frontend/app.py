import streamlit as st
import requests
import os
import json
import datetime

# Page Config
st.set_page_config(
    page_title="DocAnalyzer | Document Analyzer & Retriever",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")

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
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant that answers questions based on the given documents. Provide accurate and concise answers and cite the page numbers."
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# --- LEFT SIDEBAR (NAV & UPLOADED DOCS) ---
with st.sidebar:
    # Logo & App Title
    col_icon, col_txt = st.columns([0.25, 0.75])
    with col_icon:
        st.markdown("<h2 style='margin:0; color:#5850EC;'>📄</h2>", unsafe_allow_html=True)
    with col_txt:
        st.markdown("<h3 style='margin:0; font-weight:700; color:#0F172A;'>DocAnalyzer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; font-size:0.75rem; color:#64748B;'>Document Analyzer & Retriever</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Menu
    nav = st.radio(
        "Navigation",
        ["💬 Chat", "📥 Upload Document", "⚙️ System Prompt", "🕒 History"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#5850EC; margin-bottom:8px;'>Uploaded Documents</p>", unsafe_allow_html=True)

    # Active Uploaded Document Card
    if st.session_state.current_filename:
        meta = st.session_state.doc_metadata or {}
        pages = meta.get("total_pages", "12")
        words = meta.get("total_words", 0)
        size_mb = round(meta.get("total_chars", 1200000) / 1000000, 1)

        st.markdown(f"""
        <div class="doc-item-card">
            <div>
                <p class="doc-item-title">📄 {st.session_state.current_filename}</p>
                <p class="doc-item-meta">{pages} pages • {size_mb if size_mb > 0 else 1.2} MB</p>
            </div>
            <span class="status-badge-green">✔</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No document loaded yet.")

    # Upload New Document Button / File Uploader
    with st.expander("+ Upload New Document", expanded=(nav == "📥 Upload Document")):
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Submit & Index PDF", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        up_resp = requests.post(f"{BACKEND_URL}/api/upload", files=files, timeout=120)
                        if up_resp.status_code == 200:
                            up_data = up_resp.json()
                            st.session_state.current_filename = up_data["filename"]
                            st.session_state.doc_metadata = up_data["metadata"]

                            # Process chunking
                            proc_resp = requests.post(f"{BACKEND_URL}/api/process", json={
                                "filename": up_data["filename"],
                                "strategy": "recursive",
                                "chunk_size": 500,
                                "chunk_overlap": 50
                            }, timeout=120)
                            if proc_resp.status_code == 200:
                                st.success(f"'{up_data['filename']}' indexed successfully!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Upload error: {e}")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#94a3b8; text-align:center;'>Made with ❤️ by Devika</p>", unsafe_allow_html=True)

# --- MAIN CONTENT AREA (MIDDLE + RIGHT COLUMNS) ---
main_col, right_col = st.columns([2.5, 1])

# --- MIDDLE COLUMN: CHAT & RETRIEVAL ---
with main_col:
    # Header Title Bar
    h_col1, h_col2 = st.columns([0.8, 0.2])
    with h_col1:
        st.markdown("<h2 class='doc-header-title'>Document Analyzer & Retriever</h2>", unsafe_allow_html=True)
        st.markdown("<p class='doc-header-sub'>Ask anything about your documents</p>", unsafe_allow_html=True)
    with h_col2:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Render Chat History Messages
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Upload a PDF document and ask any question to start analyzing!")
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                sources = msg.get("sources", [])
                timestamp = msg.get("timestamp", datetime.datetime.now().strftime("%I:%M %p"))

                if role == "user":
                    st.markdown(f"""
                    <div class="chat-row-user">
                        <div style="display:flex; flex-direction:column; align-items:flex-end;">
                            <div class="chat-bubble-user">{content}</div>
                            <div class="chat-timestamp">{timestamp}</div>
                        </div>
                        <div class="chat-avatar-user">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sources_html = ""
                    if sources:
                        for s in sources[:2]:
                            pg = s.get("page_number", 1)
                            sources_html += f"""<div class="source-pill">📄 Source: Page {pg}</div> """

                    st.markdown(f"""
                    <div class="chat-row-bot">
                        <div class="chat-avatar-bot">🤖</div>
                        <div style="width:100%;">
                            <div class="chat-bubble-bot">
                                <div>{content}</div>
                                {f'<div style="margin-top:8px;">{sources_html}</div>' if sources_html else ''}
                            </div>
                            <div class="chat-timestamp">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat Input Section at Bottom
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask a question about your document...")

    if user_input:
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": now_str
        })

        # Submit to RAG Engine
        try:
            chat_payload = {
                "session_id": st.session_state.session_id,
                "query": user_input,
                "filename": st.session_state.current_filename,
                "system_prompt": st.session_state.system_prompt,
                "top_k": st.session_state.top_k,
                "temperature": 0.2
            }
            resp = requests.post(f"{BACKEND_URL}/api/chat", json=chat_payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error from backend: {resp.text}",
                    "sources": [],
                    "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Connection error: {e}",
                "sources": [],
                "timestamp": datetime.datetime.now().strftime("%I:%M %p")
            })

        st.rerun()

    st.markdown("<p class='disclaimer-text'>Answers are generated using AI and may not always be 100% accurate.</p>", unsafe_allow_html=True)

# --- RIGHT COLUMN: CONTROLS & SYSTEM PROMPT ---
with right_col:
    # 1. System Prompt Box
    st.markdown("""
    <div class="custom-card">
        <p class="card-title">⚙️ System Prompt</p>
        <p class="card-sub">Customize the system prompt</p>
    </div>
    """, unsafe_allow_html=True)

    new_sys_prompt = st.text_area(
        "System Prompt Area",
        value=st.session_state.system_prompt,
        height=180,
        label_visibility="collapsed"
    )

    if st.button("Update Prompt", use_container_width=True, key="update_prompt_btn"):
        st.session_state.system_prompt = new_sys_prompt
        st.success("System prompt updated!")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Model Card
    st.markdown("""
    <div class="custom-card">
        <p class="card-title">⚙️ Model</p>
        <p style="font-size:0.88rem; font-weight:600; color:#1e293b; margin-bottom:6px;">Cloudflare Workers AI</p>
        <span class="status-badge-green">🟢 Online</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Top K Results Selector
    st.markdown("""
    <div class="custom-card">
        <p class="card-title">Top K Results</p>
    </div>
    """, unsafe_allow_html=True)

    top_k_val = st.selectbox(
        "Top K Results Select",
        [1, 2, 3, 4, 5, 8, 10],
        index=4,
        label_visibility="collapsed"
    )
    st.session_state.top_k = top_k_val
