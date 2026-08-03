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

# Session State & Permanent Auto-Login Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an expert AI Document Assistant. Provide accurate, detailed explanations based strictly on the provided document context."
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "💬 Chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 8

# Auto-Remember Login via Query Params
query_params = st.query_params
if not st.session_state.authenticated:
    remembered_user = query_params.get("user") or query_params.get("email")
    if remembered_user and "@" in remembered_user:
        st.session_state.authenticated = True
        st.session_state.user_email = remembered_user
        st.session_state.user_name = remembered_user.split("@")[0].capitalize()

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
            if "@" in g_email and "." in g_email:
                try:
                    v_resp = requests.post(f"{BACKEND_URL}/api/auth/verify", json={"token_or_email": g_email}, timeout=5)
                    if v_resp.status_code == 200:
                        u_info = v_resp.json()["user"]
                        st.session_state.authenticated = True
                        st.session_state.user_email = u_info["email"]
                        st.session_state.user_name = u_info["name"]
                        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
                        st.query_params["user"] = u_info["email"]
                        st.success(f"Welcome back, {u_info['name']}!")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Please check your email.")
                except Exception as e:
                    st.error(f"Auth error: {e}")
            else:
                st.warning("Please enter a valid Google Email address.")

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

    # Upload New Document Button
    with st.expander("+ Upload New PDF Document", expanded=(st.session_state.active_nav == "📥 Upload Document")):
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Submit & Index PDF", use_container_width=True):
                with st.spinner("Uploading to Storage Bucket & Indexing..."):
                    try:
                        target_backend = get_backend_url()
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"user_id": st.session_state.user_email}
                        up_resp = requests.post(f"{target_backend}/api/upload", files=files, data=data, timeout=180)
                        if up_resp.status_code == 200:
                            up_data = up_resp.json()
                            st.session_state.current_filename = up_data["filename"]
                            st.session_state.doc_metadata = up_data["metadata"]

                            # Process chunking
                            proc_resp = requests.post(f"{target_backend}/api/process", json={
                                "filename": up_data["filename"],
                                "user_id": st.session_state.user_email,
                                "strategy": "recursive",
                                "chunk_size": 500,
                                "chunk_overlap": 50
                            }, timeout=180)
                            if proc_resp.status_code == 200:
                                b_type = up_data.get("bucket_info", {}).get("storage_type", "Storage Bucket")
                                st.success(f"'{up_data['filename']}' saved and indexed!")
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
        if st.session_state.current_filename:
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
                    "user_id": st.session_state.user_email,
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

    # VIEW 2: MY STORAGE BUCKET FILES
    elif st.session_state.active_nav == "📦 My Storage Bucket Files":
        st.subheader("📦 My Uploaded Documents")
        st.caption(f"Showing documents uploaded by User `{st.session_state.user_email}`")

        try:
            b_resp = requests.get(f"{BACKEND_URL}/api/bucket/files?user_id={st.session_state.user_email}", timeout=10)
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
                            <span class="status-badge-green">✔ Active</span>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if st.button(f"🚀 Set Active & Analyze '{f['filename']}'", key=f"load_file_{f['filename']}"):
                                with st.spinner(f"Indexing '{f['filename']}'..."):
                                    p_resp = requests.post(f"{BACKEND_URL}/api/process", json={
                                        "filename": f['filename'],
                                        "user_id": st.session_state.user_email,
                                        "strategy": "recursive",
                                        "chunk_size": 500,
                                        "chunk_overlap": 50
                                    }, timeout=180)
                                    if p_resp.status_code == 200:
                                        st.session_state.current_filename = f['filename']
                                        st.session_state.active_nav = "💬 Chat"
                                        st.success(f"'{f['filename']}' set as active document!")
                                        st.rerun()
                        with c2:
                            if st.button(f"🗑️ Delete from Bucket", key=f"del_file_{f['filename']}"):
                                requests.delete(f"{BACKEND_URL}/api/bucket/files/{f['filename']}", timeout=10)
                                st.success(f"Deleted '{f['filename']}'!")
                                st.rerun()
                else:
                    st.info("You haven't uploaded any documents yet. Use '+ Upload New PDF Document' in the sidebar.")
        except Exception as e:
            st.error(f"Error reading Storage Bucket: {e}")

    # VIEW 3: MY SAVED CHAT HISTORY
    elif st.session_state.active_nav == "🕒 My Saved Chat History":
        st.subheader("🕒 My Saved Chat History")
        st.caption(f"Showing chat history for User `{st.session_state.user_email}`")

        try:
            resp = requests.get(f"{BACKEND_URL}/api/sessions?user_id={st.session_state.user_email}", timeout=10)
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
                                        st.success("Conversation restored!")
                                        st.rerun()
                            with col_b:
                                if st.button(f"🗑️ Delete History", key=f"hist_del_{s['session_id']}"):
                                    requests.delete(f"{BACKEND_URL}/api/sessions/{s['session_id']}", timeout=10)
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
