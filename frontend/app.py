import streamlit as st
import requests
import os
import json
import uuid
import datetime

# Page Config
st.set_page_config(
    page_title="DocAnalyzer | Document Analyzer & Retriever",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Priority Internal Port 8001 Resolver
def resolve_working_backend_url() -> str:
    if "cached_backend_url" in st.session_state and st.session_state.cached_backend_url:
        try:
            r = requests.get(f"{st.session_state.cached_backend_url}/api/health", timeout=1)
            if r.status_code == 200 and isinstance(r.json(), dict) and r.json().get("service") == "DocAnalyzer API":
                return st.session_state.cached_backend_url
        except Exception:
            pass

    backend_port = os.getenv("BACKEND_PORT", "8001")
    env_backend = os.getenv("BACKEND_URL", "").rstrip("/")

    # Priority 1: Check internal local container port 8001 where FastAPI runs
    local_candidates = [
        "http://127.0.0.1:8001",
        f"http://127.0.0.1:{backend_port}",
        "http://0.0.0.0:8001",
        "http://localhost:8001",
        f"http://0.0.0.0:{backend_port}",
        f"http://localhost:{backend_port}",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000"
    ]

    if env_backend and env_backend not in local_candidates:
        local_candidates.insert(0, env_backend)

    for candidate in local_candidates:
        try:
            r = requests.get(f"{candidate}/api/health", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and data.get("service") == "DocAnalyzer API":
                    st.session_state.cached_backend_url = candidate
                    return candidate
        except Exception:
            pass

    fallback = "http://127.0.0.1:8001"
    st.session_state.cached_backend_url = fallback
    return fallback

BACKEND_URL = resolve_working_backend_url()

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State & Permanent Auto-Login Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an expert AI Document Assistant. Provide direct, accurate, detail-specific explanations based strictly on the provided documents."
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "💬 Chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 8
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# PERMANENT BROWSER AUTO-LOGIN
query_params = st.query_params
if not st.session_state.authenticated:
    param_user = query_params.get("user") or query_params.get("email")
    if param_user and "@" in param_user:
        clean_p = param_user.strip().lower()
        st.session_state.authenticated = True
        st.session_state.user_email = clean_p
        st.session_state.user_name = clean_p.split("@")[0].replace(".", " ").title()

# AUTO-RESTORE USER RECENT DATA & CHAT HISTORY ON REFRESH
if st.session_state.authenticated and not st.session_state.data_loaded:
    try:
        b_url = resolve_working_backend_url()
        s_resp = requests.get(f"{b_url}/api/sessions?user_id={st.session_state.user_email}", timeout=5)
        if s_resp.status_code == 200:
            sessions = s_resp.json()
            if sessions:
                latest = sessions[0]
                target_id = latest["session_id"]
                sess_resp = requests.get(f"{b_url}/api/sessions/{target_id}", timeout=5)
                if sess_resp.status_code == 200:
                    sess_data = sess_resp.json()
                    st.session_state.session_id = target_id
                    st.session_state.current_filename = sess_data["session"]["filename"]
                    st.session_state.system_prompt = sess_data["session"]["system_prompt"]
                    st.session_state.messages = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                            "sources": m.get("sources", []),
                            "timestamp": m.get("timestamp", datetime.datetime.now().strftime("%I:%M %p"))
                        }
                        for m in sess_data["messages"]
                    ]
        st.session_state.data_loaded = True
    except Exception:
        pass

# --- FIREBASE GOOGLE SIGN-IN SCREEN ---
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05); text-align: center;">
            <h2 style="color: #4F46E5; margin-bottom: 4px; font-weight:700;">📄 DocAnalyzer AI</h2>
            <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 24px;">Sign in with Google to access your documents & chat history</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.88rem; color:#475569; text-align:center;'>Enter your Google Email address to sign in:</p>", unsafe_allow_html=True)
        g_email = st.text_input("Google Email Address", placeholder="user@gmail.com", key="g_email_input", label_visibility="collapsed")

        if st.button("🔥 Continue with Google Sign-In", use_container_width=True, type="primary"):
            clean_email = g_email.strip().lower()
            if "@" in clean_email and "." in clean_email.split("@")[-1]:
                b_url = resolve_working_backend_url()
                try:
                    v_resp = requests.post(f"{b_url}/api/auth/verify", json={"token_or_email": clean_email}, timeout=5)
                    if v_resp.status_code == 200:
                        u_info = v_resp.json().get("user", {})
                        st.session_state.user_name = u_info.get("name", clean_email.split("@")[0].title())
                    else:
                        st.session_state.user_name = clean_email.split("@")[0].title()
                except Exception:
                    st.session_state.user_name = clean_email.split("@")[0].title()

                st.session_state.authenticated = True
                st.session_state.user_email = clean_email
                st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
                st.query_params["user"] = clean_email
                st.success(f"Welcome back, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.warning("Please enter a valid email address (e.g. user@gmail.com).")

    st.stop()

# --- MAIN DASHBOARD ---

# --- LEFT SIDEBAR (NAV & USER BADGE) ---
with st.sidebar:
    col_icon, col_txt = st.columns([0.25, 0.75])
    with col_icon:
        st.markdown("<h2 style='margin:0; color:#5850EC;'>📄</h2>", unsafe_allow_html=True)
    with col_txt:
        st.markdown("<h3 style='margin:0; font-weight:700; color:#0F172A;'>DocAnalyzer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; font-size:0.75rem; color:#64748B;'>Document Analyzer & Retriever</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # User Profile Badge
    st.markdown(f"""
    <div style="background:#F1F5F9; border-radius:10px; padding:12px; border:1px solid #CBD5E1;">
        <p style="margin:0; font-size:0.75rem; color:#64748B; font-weight:600;">LOGGED IN USER</p>
        <p style="margin:2px 0 0 0; font-size:0.88rem; color:#0F172A; font-weight:700; word-break:break-all;">🟢 {st.session_state.user_email}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.session_state.messages = []
        st.session_state.current_filename = None
        st.session_state.active_documents = []
        st.session_state.data_loaded = False
        st.query_params.clear()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Menu
    st.session_state.active_nav = st.radio(
        "Navigation",
        ["💬 Chat", "📥 Upload Document", "📦 My Storage Bucket Files", "🕒 My Saved Chat History"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload New Document Section with Multi-Document Choices
    with st.expander("+ Upload PDF Document", expanded=(st.session_state.active_nav == "📥 Upload Document")):
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#1E293B; margin-top:8px;'>Upload Action:</p>", unsafe_allow_html=True)
        upload_action = st.radio(
            "Upload Action Radio",
            ["➕ Add Document to Current Conversation", "💬 Start New Chat with this Document"],
            key="upload_action_choice",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("Submit & Index PDF", use_container_width=True):
                with st.spinner("Uploading to Storage Bucket & Indexing..."):
                    try:
                        b_url = resolve_working_backend_url()
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"user_id": st.session_state.user_email}
                        up_resp = requests.post(f"{b_url}/api/upload", files=files, data=data, timeout=180)
                        if up_resp.status_code == 200:
                            up_data = up_resp.json()
                            fn = up_data["filename"]

                            # Process chunking
                            proc_resp = requests.post(f"{b_url}/api/process", json={
                                "filename": fn,
                                "user_id": st.session_state.user_email,
                                "strategy": "recursive",
                                "chunk_size": 500,
                                "chunk_overlap": 50
                            }, timeout=180)

                            if proc_resp.status_code == 200:
                                if upload_action == "💬 Start New Chat with this Document":
                                    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
                                    st.session_state.messages = []
                                    st.session_state.current_filename = fn
                                    st.session_state.active_documents = [fn]
                                    st.success(f"Started new chat with '{fn}'!")
                                else:
                                    # Add to current conversation
                                    if fn not in st.session_state.active_documents:
                                        st.session_state.active_documents.append(fn)
                                    st.session_state.current_filename = fn
                                    st.success(f"Added '{fn}' to current conversation!")

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
            st.rerun()

    st.divider()

    # VIEW 1: MAIN CHAT VIEW
    if st.session_state.active_nav == "💬 Chat" or st.session_state.active_nav == "📥 Upload Document":
        if st.session_state.active_documents:
            docs_str = ", ".join([f"**{d}**" for d in st.session_state.active_documents])
            st.info(f"📁 Active Documents in Conversation: {docs_str}")
        elif st.session_state.current_filename:
            st.info(f"📁 Active Document: **{st.session_state.current_filename}**")
        else:
            st.warning("⚠️ No document selected. Upload a PDF or select one from 'My Storage Bucket Files'.")

        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 Upload a PDF document and ask any question to start analyzing!")
            else:
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    sources = msg.get("sources", [])

                    if role == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(content)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(content)
                            if sources:
                                sources_html = "".join([
                                    f"""<span class="source-pill">📄 Page {s.get('page_number', 1)}</span> """
                                    for s in sources[:4]
                                ])
                                st.markdown(f"<div style='margin-top:10px;'>{sources_html}</div>", unsafe_allow_html=True)

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
                b_url = resolve_working_backend_url()
                chat_payload = {
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_email,
                    "query": user_input,
                    "filename": st.session_state.current_filename,
                    "system_prompt": st.session_state.system_prompt,
                    "top_k": st.session_state.top_k,
                    "temperature": 0.1
                }
                resp = requests.post(f"{b_url}/api/chat", json=chat_payload, timeout=90)
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

    # VIEW 2: MY STORAGE BUCKET FILES
    elif st.session_state.active_nav == "📦 My Storage Bucket Files":
        st.subheader("📦 My Uploaded Documents")
        st.caption(f"Showing documents uploaded by User `{st.session_state.user_email}`")

        try:
            b_url = resolve_working_backend_url()
            b_resp = requests.get(f"{b_url}/api/bucket/files?user_id={st.session_state.user_email}", timeout=10)
            if b_resp.status_code == 200:
                b_data = b_resp.json()
                b_name = b_data.get("bucket_name", "Storage Bucket")
                files = b_data.get("files", [])

                if files:
                    for f in files:
                        size_mb = round(f['size_bytes'] / 1000000, 2)
                        st.markdown(f"""
                        <div class="doc-item-card">
                            <div>
                                <p class="doc-item-title">📄 {f['filename']}</p>
                                <p class="doc-item-meta">{size_mb} MB • Owner: {f.get('user_id', 'User')}</p>
                            </div>
                            <span class="status-badge-green">✔ Saved in Bucket</span>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1:
                            if st.button(f"🚀 Set Active '{f['filename']}'", key=f"load_file_{f['filename']}"):
                                with st.spinner(f"Indexing '{f['filename']}'..."):
                                    p_resp = requests.post(f"{b_url}/api/process", json={
                                        "filename": f['filename'],
                                        "user_id": st.session_state.user_email,
                                        "strategy": "recursive",
                                        "chunk_size": 500,
                                        "chunk_overlap": 50
                                    }, timeout=180)
                                    if p_resp.status_code == 200:
                                        st.session_state.current_filename = f['filename']
                                        if f['filename'] not in st.session_state.active_documents:
                                            st.session_state.active_documents.append(f['filename'])
                                        st.session_state.active_nav = "💬 Chat"
                                        st.success(f"'{f['filename']}' set as active document!")
                                        st.rerun()
                        with c2:
                            if st.button(f"➕ Add to Conversation", key=f"add_conv_{f['filename']}"):
                                if f['filename'] not in st.session_state.active_documents:
                                    st.session_state.active_documents.append(f['filename'])
                                st.session_state.current_filename = f['filename']
                                st.session_state.active_nav = "💬 Chat"
                                st.success(f"Added '{f['filename']}' to conversation!")
                                st.rerun()
                        with c3:
                            if st.button(f"🗑️ Delete from Bucket", key=f"del_file_{f['filename']}"):
                                requests.delete(f"{b_url}/api/bucket/files/{f['filename']}", timeout=10)
                                if f['filename'] in st.session_state.active_documents:
                                    st.session_state.active_documents.remove(f['filename'])
                                st.success(f"Deleted '{f['filename']}'!")
                                st.rerun()
                else:
                    st.info("You haven't uploaded any documents yet. Use '+ Upload PDF Document' in the sidebar.")
        except Exception as e:
            st.error(f"Error reading Storage Bucket: {e}")

    # VIEW 3: MY SAVED CHAT HISTORY
    elif st.session_state.active_nav == "🕒 My Saved Chat History":
        st.subheader("🕒 My Saved Chat History")
        st.caption(f"Showing chat history for User `{st.session_state.user_email}`")

        try:
            b_url = resolve_working_backend_url()
            resp = requests.get(f"{b_url}/api/sessions?user_id={st.session_state.user_email}", timeout=10)
            if resp.status_code == 200:
                sessions = resp.json()
                if sessions:
                    st.caption(f"Total Saved Conversations: `{len(sessions)}`")
                    for s in sessions:
                        with st.expander(f"💬 Conversation | Document: {s['filename']} | {s['message_count']} msgs"):
                            st.markdown(f"**Created At:** `{s['created_at']}`")
                            st.markdown(f"**System Prompt:** *{s['system_prompt']}*")

                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                if st.button(f"📥 Restore Conversation", key=f"hist_load_{s['session_id']}"):
                                    sess_resp = requests.get(f"{b_url}/api/sessions/{s['session_id']}", timeout=10)
                                    if sess_resp.status_code == 200:
                                        sess_data = sess_resp.json()
                                        st.session_state.session_id = s['session_id']
                                        st.session_state.current_filename = sess_data["session"]["filename"]
                                        st.session_state.active_documents = [sess_data["session"]["filename"]]
                                        st.session_state.system_prompt = sess_data["session"]["system_prompt"]
                                        st.session_state.messages = [
                                            {
                                                "role": m["role"],
                                                "content": m["content"],
                                                "sources": m.get("sources", []),
                                                "timestamp": m.get("timestamp", datetime.datetime.now().strftime("%I:%M %p"))
                                            }
                                            for m in sess_data["messages"]
                                        ]
                                        st.session_state.active_nav = "💬 Chat"
                                        st.success("Conversation restored!")
                                        st.rerun()
                            with col_b:
                                if st.button(f"🗑️ Delete History", key=f"hist_del_{s['session_id']}"):
                                    requests.delete(f"{b_url}/api/sessions/{s['session_id']}", timeout=10)
                                    st.success("Conversation deleted.")
                                    st.rerun()
                else:
                    st.info("No saved chat history found.")
        except Exception as e:
            st.error(f"Failed to fetch chat history: {e}")

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
