import os
from pathlib import Path
from dotenv import load_dotenv

# Disable ChromaDB anonymous telemetry logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

def get_clean_env(key: str, default: str = "") -> str:
    val = os.getenv(key, default).strip()
    if not val or val.lower() in ["none", "null", "undefined"]:
        return ""
    return val

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Detect Railway Persistent Volume Storage Mount Path
env_data_dir = get_clean_env("DATA_DIR") or get_clean_env("RAILWAY_VOLUME_MOUNT_PATH")
if env_data_dir:
    DATA_DIR = Path(env_data_dir)
elif Path("/data").exists():
    DATA_DIR = Path("/data")
else:
    DATA_DIR = Path("/app/storage")

DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/app/storage")
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
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))

# Default System Prompt presets
DEFAULT_SYSTEM_PROMPT = """You are a precise, highly analytical Document AI assistant.
Your goal is to answer the user's questions strictly based on the provided document context.
If the answer cannot be found in the context, explicitly state that the document does not contain that information.
Always cite section titles or page numbers when quoting facts."""
