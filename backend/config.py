import os
import sys
from pathlib import Path

# Insert project root to sys.path before any relative package imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

# Disable ChromaDB anonymous telemetry logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

def get_clean_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default).strip()
    if not val or val.lower() in ["none", "null", "undefined"]:
        return ""
    return val

def get_safe_data_dir() -> Path:
    """
    Safely resolves a writable DATA_DIR path without permission crashes.
    """
    env_dir = get_clean_env("DATA_DIR") or get_clean_env("RAILWAY_VOLUME_MOUNT_PATH")
    if env_dir:
        try:
            p = Path(env_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass

    if Path("/data").exists():
        try:
            p = Path("/data")
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass

    # Guaranteed Writable Fallback inside App Repository
    fallback = BASE_DIR / "storage"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    except Exception:
        # Ultimate temporary fallback
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "doc_analyser_storage"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

DATA_DIR = get_safe_data_dir()

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_DATA_DIR = DATA_DIR / "backup"
BACKUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_DB_DIR = DATA_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DB_PATH = DATA_DIR / "chat_history.db"

# Cloudflare Workers AI settings
CLOUDFLARE_ACCOUNT_ID = get_clean_env("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = get_clean_env("CLOUDFLARE_API_TOKEN")

# Models
CLOUDFLARE_EMBEDDING_MODEL = os.getenv(
    "CLOUDFLARE_EMBEDDING_MODEL", "@cf/baai/bge-large-en-v1.5"
)
CLOUDFLARE_LLM_MODEL = os.getenv(
    "CLOUDFLARE_LLM_MODEL", "@cf/meta/llama-3.1-8b-instruct"
)

# API Host & Port
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8001))

# Default System Prompt presets
DEFAULT_SYSTEM_PROMPT = """You are a precise, highly analytical Document AI assistant.
Your goal is to answer the user's questions strictly based on the provided document context.
If the answer cannot be found in the context, explicitly state that the document does not contain that information.
Always cite section titles or page numbers when quoting facts."""
