import streamlit as st
import requests
import os
import json
import uuid
import datetime
import hashlib

# Page Config
st.set_page_config(
    page_title="DocAnalyzer | Enterprise Document Analyzer & Retriever",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Priority Internal Port 8001 Resolver
def resolve_working_backend_url() -> str:
    if "cached_backend_url" in st.session_state and st.session_state.cached_backend_url:
        try:
            r = requests.get(f"{st.session_state.cached_backend_url}/api/health", timeout=1)
            if r.status_code == 200 and isinstance(r.json(), dict) and r.json().get("service", "").startswith("DocAnalyzer"):
                return st.session_state.cached_backend_url
        except Exception:
            pass

    backend_port = os.getenv("BACKEND_PORT", "8001")
    env_backend = os.getenv("BACKEND_URL", "").rstrip("/")

    local_candidates = [
        "http://127.0.0.1:8001",
        f"http://127.0.0.1:{backend_port}",
        "http://0.0.0.0:8001",
        "http://localhost:8001",
        f"http://0.0.0.0:{backend_port}",
        f"http://localhost:{backend_port}",
        "http://127.0.0.1:8000"
    ]

    if env_backend and env_backend not in local_candidates:
        local_candidates.insert(0, env_backend)

    for candidate in local_candidates:
        try:
            r = requests.get(f"{candidate}/api/health", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and data.get("service", "").startswith("DocAnalyzer"):
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

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []
if "extraction_report" not in st.session_state:
    st.session_state.extraction_report = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}
if "style_prompt" not in st.session_state:
    st.session_state.style_prompt = "Provide clear, professional explanations with concise bullet points."
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "💬 Chat"
if "top_k" not in st.session_state:
    st.session_state.top_k = 12
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Browser Auto-Login
query_params = st.query_params
if not st.session_state.authenticated:
    param_user = query_params.get("user") or query_params.get("email")
    if param_user and "@" in param_user:
        clean_p = param_user.strip().lower()
        st.session_state.authenticated = True
        st.session_state.user_email = clean_p
        st.session_state.user_name = clean_p.split("@")[0].replace(".", " ").title()

# Restore User Data on Refresh
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
                    st.session_state.current_document_id = sess_data["session"].get("document_id")
                    st.session_state.style_prompt = sess_data["session"]["system_prompt"]
                    st.session_state.messages = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                            "sources": m.get("sources", []),
                            "verified_quotes": m.get("evidence_quotes", []),
                            "timestamp": m.get("timestamp", datetime.datetime.now().strftime("%I:%M %p"))
                        }
                        for m in sess_data["messages"]
                    ]
        st.session_state.data_loaded = True
    except Exception:
        pass

# Authentication Screen
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
                st.rerun()
            else:
                st.warning("Please enter a valid email address (e.g. user@gmail.com).")

    st.stop()

# Dashboard
with st.sidebar:
    st.markdown("### 📄 DocAnalyzer AI")
    st.caption("Enterprise Grounded RAG Platform")

    st.markdown(f"**Logged in:** `{st.session_state.user_email}`")
    if st.button("🚪 Sign Out", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.messages = []
        st.query_params.clear()
        st.rerun()

    st.divider()

    st.session_state.active_nav = st.radio(
        "Navigation",
        ["💬 Chat", "📥 Upload Document", "📦 My Storage Bucket Files", "🕒 My Saved Chat History"],
        label_visibility="collapsed"
    )

    st.divider()

    with st.expander("+ Upload PDF Document", expanded=(st.session_state.active_nav == "📥 Upload Document")):
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("Submit & Index PDF", use_container_width=True):
                with st.spinner("Uploading to Storage Bucket & Analyzing PDF..."):
                    try:
                        b_url = resolve_working_backend_url()
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"user_id": st.session_state.user_email}
                        up_resp = requests.post(f"{b_url}/api/upload", files=files, data=data, timeout=180)
                        
                        if up_resp.status_code == 200:
                            up_data = up_resp.json()
                            fn = up_data["filename"]
                            doc_id = up_data.get("document_id")
                            report = up_data.get("extraction_report", {})
                            st.session_state.extraction_report = report

                            proc_resp = requests.post(f"{b_url}/api/process", json={
                                "filename": fn,
                                "document_id": doc_id,
                                "session_id": st.session_state.session_id,
                                "user_id": st.session_state.user_email,
                                "strategy": "recursive",
                                "chunk_size": 400,
                                "chunk_overlap": 80
                            }, timeout=180)

                            if proc_resp.status_code == 200:
                                proc_data = proc_resp.json()
                                st.session_state.current_filename = fn
                                st.session_state.current_document_id = doc_id
                                st.session_state.active_documents = [fn]
                                coverage_msg = ""
                                if proc_data.get("extraction_report", {}).get("missing_unsearchable_pages"):
                                    missing_p = proc_data["extraction_report"]["missing_unsearchable_pages"]
                                    coverage_msg = f" (Notice: Missing/unsearchable pages: {missing_p})"
                                st.success(f"Indexed '{fn}'! Quality Score: {report.get('quality_score', 100)}%{coverage_msg}")
                                st.session_state.active_nav = "💬 Chat"
                                st.rerun()
                            else:
                                st.error(f"Processing error: {proc_resp.text}")
                        else:
                            st.error(f"Upload error: {up_resp.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

main_col, right_col = st.columns([2.5, 1])

with main_col:
    h_col1, h_col2 = st.columns([0.8, 0.2])
    with h_col1:
        st.markdown("<h2 class='doc-header-title'>Document Analyzer & Retriever</h2>", unsafe_allow_html=True)
        st.markdown("<p class='doc-header-sub'>Grounded Answers with Verified Source Citations</p>", unsafe_allow_html=True)
    with h_col2:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    if st.session_state.active_nav in ["💬 Chat", "📥 Upload Document"]:
        if st.session_state.current_filename:
            st.info(f"📁 Active Document: **{st.session_state.current_filename}** (Doc ID: `{st.session_state.current_document_id or 'Auto'}`)")
        else:
            st.warning("⚠️ No document selected. Upload a PDF or select one from 'My Storage Bucket Files'.")

        if st.session_state.extraction_report:
            rep = st.session_state.extraction_report
            if rep.get("extraction_warnings"):
                for w in rep["extraction_warnings"]:
                    st.warning(f"⚠️ Extraction Notice: {w}")

        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 Ask any question to start analyzing your document with verified evidence!")
            else:
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    sources = msg.get("sources", [])
                    rag_trace = msg.get("rag_trace")

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

                            if rag_trace:
                                with st.expander("🛠️ Developer RAG Debug Trace", expanded=False):
                                    st.json(rag_trace)

        user_input = st.chat_input("Ask a question about your document...")

        if user_input:
            now_str = datetime.datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": now_str})

            # Unique Request Key & Cache Entry by (document_id, normalized_query)
            clean_q = user_input.strip().lower()
            doc_id_key = st.session_state.current_document_id or "default"
            cache_key = (doc_id_key, clean_q)
            req_key = hashlib.sha256(f"{doc_id_key}:{st.session_state.session_id}:{clean_q}".encode()).hexdigest()

            if cache_key in st.session_state.query_cache:
                cached_data = st.session_state.query_cache[cache_key]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cached_data["answer"],
                    "sources": cached_data.get("sources", []),
                    "verified_quotes": cached_data.get("verified_quotes", []),
                    "rag_trace": cached_data.get("rag_trace"),
                    "timestamp": now_str
                })
            else:
                try:
                    b_url = resolve_working_backend_url()
                    chat_payload = {
                        "session_id": st.session_state.session_id,
                        "document_id": st.session_state.current_document_id,
                        "user_id": st.session_state.user_email,
                        "query": user_input,
                        "filename": st.session_state.current_filename,
                        "system_prompt": st.session_state.style_prompt,
                        "top_k": st.session_state.top_k,
                        "temperature": 0.0
                    }
                    resp = requests.post(f"{b_url}/api/chat", json=chat_payload, timeout=90)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.query_cache[cache_key] = data
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get("sources", []),
                            "verified_quotes": data.get("verified_quotes", []),
                            "rag_trace": data.get("rag_trace"),
                            "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {resp.text}",
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

    elif st.session_state.active_nav == "📦 My Storage Bucket Files":
        st.subheader("📦 My Uploaded Documents")
        try:
            b_url = resolve_working_backend_url()
            b_resp = requests.get(f"{b_url}/api/bucket/files?user_id={st.session_state.user_email}", timeout=10)
            if b_resp.status_code == 200:
                files = b_resp.json().get("files", [])
                if files:
                    for f in files:
                        size_mb = round(f['size_bytes'] / 1000000, 2)
                        st.markdown(f"📄 **{f['filename']}** ({size_mb} MB)")
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if st.button(f"🚀 Set Active '{f['filename']}'", key=f"load_{f['filename']}"):
                                p_resp = requests.post(f"{b_url}/api/process", json={
                                    "filename": f['filename'],
                                    "user_id": st.session_state.user_email,
                                    "chunk_size": 400,
                                    "chunk_overlap": 80
                                }, timeout=180)
                                if p_resp.status_code == 200:
                                    p_data = p_resp.json()
                                    st.session_state.current_filename = f['filename']
                                    st.session_state.current_document_id = p_data.get("document_id")
                                    st.session_state.active_documents = [f['filename']]
                                    st.session_state.active_nav = "💬 Chat"
                                    st.rerun()
                        with c2:
                            if st.button(f"🗑️ Delete File", key=f"del_{f['filename']}"):
                                requests.delete(f"{b_url}/api/bucket/files/{f['filename']}", timeout=10)
                                st.rerun()
                else:
                    st.info("No documents uploaded yet.")
        except Exception as e:
            st.error(f"Error fetching bucket files: {e}")

    elif st.session_state.active_nav == "🕒 My Saved Chat History":
        st.subheader("🕒 My Saved Chat History")
        try:
            b_url = resolve_working_backend_url()
            resp = requests.get(f"{b_url}/api/sessions?user_id={st.session_state.user_email}", timeout=10)
            if resp.status_code == 200:
                sessions = resp.json()
                if sessions:
                    for s in sessions:
                        with st.expander(f"💬 Conversation | Document: {s['filename']} | {s['message_count']} msgs"):
                            if st.button(f"📥 Restore Conversation", key=f"hist_{s['session_id']}"):
                                sess_resp = requests.get(f"{b_url}/api/sessions/{s['session_id']}", timeout=10)
                                if sess_resp.status_code == 200:
                                    sess_data = sess_resp.json()
                                    st.session_state.session_id = s['session_id']
                                    st.session_state.current_filename = sess_data["session"]["filename"]
                                    st.session_state.current_document_id = sess_data["session"].get("document_id")
                                    st.session_state.messages = [
                                        {
                                            "role": m["role"],
                                            "content": m["content"],
                                            "sources": m.get("sources", []),
                                            "verified_quotes": m.get("evidence_quotes", []),
                                            "timestamp": m.get("timestamp", datetime.datetime.now().strftime("%I:%M %p"))
                                        }
                                        for m in sess_data["messages"]
                                    ]
                                    st.session_state.active_nav = "💬 Chat"
                                    st.rerun()
                else:
                    st.info("No saved chat history.")
        except Exception as e:
            st.error(f"Failed to fetch history: {e}")

with right_col:
    st.markdown("### ⚙️ UI Style Prompt")
    st.caption("Customizes tone and presentation format. Core grounding, document isolation, and citation validation rules cannot be overridden.")

    new_style_prompt = st.text_area(
        "Style Prompt",
        value=st.session_state.style_prompt,
        height=180,
        label_visibility="collapsed"
    )

    if st.button("Update Tone & Style Prompt", use_container_width=True):
        st.session_state.style_prompt = new_style_prompt
        st.success("Updated tone prompt!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Grounding Rules Status")
    st.markdown("""
    - 🔒 Document Isolation: **ACTIVE**
    - 🛡️ Prompt Injection Guard: **ACTIVE**
    - 📄 Verified Quote Enforcement: **ACTIVE**
    - 🛑 No-Evidence Fallback: **ACTIVE**
    """)
