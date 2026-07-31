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

# Dynamic Internal Backend Resolver
def get_backend_url() -> str:
    env_url = os.getenv("BACKEND_URL", "").rstrip("/")
    b_port = os.getenv("BACKEND_PORT", "8000")
    
    candidates = []
    if env_url:
        candidates.append(env_url)
        
    for port in [b_port, "8000", "8001"]:
        for host in ["127.0.0.1", "0.0.0.0", "localhost"]:
            url = f"http://{host}:{port}"
            if url not in candidates:
                candidates.append(url)

    for candidate in candidates:
        try:
            r = requests.get(f"{candidate}/api/health", timeout=1)
            if r.status_code == 200:
                return candidate
        except Exception:
            pass

    return f"http://127.0.0.1:{b_port}"

BACKEND_URL = get_backend_url()

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
    st.session_state.system_prompt = "You are a helpful AI assistant that provides accurate explanations based strictly on the provided documents."
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "💬 Chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 8
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

# Auto-Load Recent Saved Chat Session History on Startup
if not st.session_state.history_loaded:
    try:
        s_resp = requests.get(f"{BACKEND_URL}/api/sessions", timeout=5)
        if s_resp.status_code == 200:
            sessions = s_resp.json()
            if sessions:
                latest = sessions[0]
                target_id = latest["session_id"]
                sess_resp = requests.get(f"{BACKEND_URL}/api/sessions/{target_id}", timeout=5)
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
        st.session_state.history_loaded = True
    except Exception:
        pass

# --- LEFT SIDEBAR (NAV, BUCKET FILES & SAVED SESSIONS) ---
with st.sidebar:
    col_icon, col_txt = st.columns([0.25, 0.75])
    with col_icon:
        st.markdown("<h2 style='margin:0; color:#5850EC;'>📄</h2>", unsafe_allow_html=True)
    with col_txt:
        st.markdown("<h3 style='margin:0; font-weight:700; color:#0F172A;'>DocAnalyzer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; font-size:0.75rem; color:#64748B;'>Document Analyzer & Retriever</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Health Check Status
    try:
        active_url = get_backend_url()
        h_resp = requests.get(f"{active_url}/api/health", timeout=3)
        if h_resp.status_code == 200:
            h_data = h_resp.json()
            bucket = h_data.get("bucket_name", "Storage Bucket")
            st.success(f"🟢 Connected | Bucket: `{bucket}`")
    except Exception:
        st.warning("🟠 Backend Starting...")

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Menu
    st.session_state.active_nav = st.radio(
        "Navigation",
        ["💬 Chat", "📥 Upload Document", "📦 Storage Bucket Files", "🕒 Saved Chat History"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload New Document Button
    with st.expander("+ Upload New PDF Document", expanded=(st.session_state.active_nav == "📥 Upload Document")):
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Submit & Index PDF", use_container_width=True):
                with st.spinner("Uploading to Storage Bucket & Indexing..."):
                    try:
                        target_backend = get_backend_url()
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        up_resp = requests.post(f"{target_backend}/api/upload", files=files, timeout=180)
                        if up_resp.status_code == 200:
                            up_data = up_resp.json()
                            st.session_state.current_filename = up_data["filename"]
                            st.session_state.doc_metadata = up_data["metadata"]

                            # Process chunking
                            proc_resp = requests.post(f"{target_backend}/api/process", json={
                                "filename": up_data["filename"],
                                "strategy": "recursive",
                                "chunk_size": 500,
                                "chunk_overlap": 50
                            }, timeout=180)
                            if proc_resp.status_code == 200:
                                b_type = up_data.get("bucket_info", {}).get("storage_type", "Storage Bucket")
                                st.success(f"'{up_data['filename']}' saved in '{b_type}' and indexed!")
                                st.session_state.active_nav = "💬 Chat"
                                st.rerun()
                            else:
                                st.error(f"Processing error: {proc_resp.text}")
                        else:
                            st.error(f"Upload error: {up_resp.text}")
                    except Exception as e:
                        st.error(f"Upload connection error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#94a3b8; text-align:center;'>Made with ❤️ by Devika</p>", unsafe_allow_html=True)

# --- MAIN CONTENT AREA (MIDDLE + RIGHT COLUMNS) ---
main_col, right_col = st.columns([2.5, 1])

# --- MIDDLE COLUMN: MAIN DASHBOARD VIEWS ---
with main_col:
    # Header Title Bar
    h_col1, h_col2 = st.columns([0.8, 0.2])
    with h_col1:
        st.markdown("<h2 class='doc-header-title'>Document Analyzer & Retriever</h2>", unsafe_allow_html=True)
        st.markdown("<p class='doc-header-sub'>Ask anything about your documents</p>", unsafe_allow_html=True)
    with h_col2:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

    st.divider()

    # VIEW 1: MAIN CHATVIEW
    if st.session_state.active_nav == "💬 Chat" or st.session_state.active_nav == "📥 Upload Document":
        if st.session_state.current_filename:
            st.info(f"📁 Active Document: **{st.session_state.current_filename}**")

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
                            for s in sources[:4]:
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

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.chat_input("Ask a question about your document...")

        if user_input:
            now_str = datetime.datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": now_str
            })

            try:
                target_backend = get_backend_url()
                chat_payload = {
                    "session_id": st.session_state.session_id,
                    "query": user_input,
                    "filename": st.session_state.current_filename,
                    "system_prompt": st.session_state.system_prompt,
                    "top_k": st.session_state.top_k,
                    "temperature": 0.2
                }
                resp = requests.post(f"{target_backend}/api/chat", json=chat_payload, timeout=90)
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

    # VIEW 2: STORAGE BUCKET FILES VIEWER
    elif st.session_state.active_nav == "📦 Storage Bucket Files":
        st.subheader("📦 Uploaded Storage Bucket Documents")
        try:
            b_resp = requests.get(f"{BACKEND_URL}/api/bucket/files", timeout=10)
            if b_resp.status_code == 200:
                b_data = b_resp.json()
                b_name = b_data.get("bucket_name", "Storage Bucket")
                files = b_data.get("files", [])

                st.caption(f"Active Storage Bucket: `{b_name}` | Total Uploaded PDFs: `{len(files)}`")

                if files:
                    for f in files:
                        size_mb = round(f['size_bytes'] / 1000000, 2)
                        st.markdown(f"""
                        <div class="doc-item-card">
                            <div>
                                <p class="doc-item-title">📄 {f['filename']}</p>
                                <p class="doc-item-meta">{size_mb} MB • Last Modified: {f.get('last_modified', 'N/A')}</p>
                            </div>
                            <span class="status-badge-green">✔ Stored</span>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if st.button(f"🚀 Load & Analyze '{f['filename']}'", key=f"load_file_{f['filename']}"):
                                with st.spinner(f"Indexing '{f['filename']}'..."):
                                    p_resp = requests.post(f"{BACKEND_URL}/api/process", json={
                                        "filename": f['filename'],
                                        "strategy": "recursive",
                                        "chunk_size": 500,
                                        "chunk_overlap": 50
                                    }, timeout=180)
                                    if p_resp.status_code == 200:
                                        st.session_state.current_filename = f['filename']
                                        st.session_state.active_nav = "💬 Chat"
                                        st.success(f"Loaded '{f['filename']}'!")
                                        st.rerun()
                        with c2:
                            if st.button(f"🗑️ Delete from Bucket", key=f"del_file_{f['filename']}"):
                                requests.delete(f"{BACKEND_URL}/api/bucket/files/{f['filename']}", timeout=10)
                                st.success(f"Deleted '{f['filename']}'!")
                                st.rerun()
                else:
                    st.info("No documents currently found in the Storage Bucket. Upload a PDF using the left sidebar.")
        except Exception as e:
            st.error(f"Error reading Storage Bucket: {e}")

    # VIEW 3: SAVED CHAT HISTORY BROWSER
    elif st.session_state.active_nav == "🕒 Saved Chat History":
        st.subheader("🕒 Saved Chat Sessions (Railway Volume)")
        try:
            resp = requests.get(f"{BACKEND_URL}/api/sessions", timeout=10)
            if resp.status_code == 200:
                sessions = resp.json()
                if sessions:
                    st.caption(f"Total Saved Sessions: `{len(sessions)}`")
                    for s in sessions:
                        with st.expander(f"💬 Session `{s['session_id']}` | Document: {s['filename']} | {s['message_count']} msgs"):
                            st.markdown(f"**Created At:** `{s['created_at']}`")
                            st.markdown(f"**System Prompt:** *{s['system_prompt']}*")

                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                if st.button(f"📥 Restore Session into Chat", key=f"hist_load_{s['session_id']}"):
                                    sess_resp = requests.get(f"{BACKEND_URL}/api/sessions/{s['session_id']}", timeout=10)
                                    if sess_resp.status_code == 200:
                                        sess_data = sess_resp.json()
                                        st.session_state.session_id = s['session_id']
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
                                        st.session_state.active_nav = "💬 Chat"
                                        st.success("Session restored successfully!")
                                        st.rerun()
                            with col_b:
                                if st.button(f"🗑️ Delete Session", key=f"hist_del_{s['session_id']}"):
                                    requests.delete(f"{BACKEND_URL}/api/sessions/{s['session_id']}", timeout=10)
                                    st.success("Session deleted.")
                                    st.rerun()
                else:
                    st.info("No saved chat sessions found in database.")
        except Exception as e:
            st.error(f"Failed to fetch session history: {e}")

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
        index=5,
        label_visibility="collapsed"
    )
    st.session_state.top_k = top_k_val
