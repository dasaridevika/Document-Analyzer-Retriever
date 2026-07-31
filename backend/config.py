import os
from pathlib import Path
from dotenv import load_dotenv

# Disable ChromaDB anonymous telemetry logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "storage"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_DB_DIR = DATA_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DB_PATH = DATA_DIR / "chat_history.db"

# Cloudflare Workers AI settings
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

# Models
CLOUDFLARE_EMBEDDING_MODEL = os.getenv(
    "CLOUDFLARE_EMBEDDING_MODEL", "@cf/baai/bge-large-en-v1.5"
)
CLOUDFLARE_LLM_MODEL = os.getenv(
    "CLOUDFLARE_LLM_MODEL", "@cf/meta/llama-3-8b-instruct"
)

# API Host & Port (Internal port 8001 to prevent port collision with Railway $PORT)
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8001))

# Default System Prompt presets
DEFAULT_SYSTEM_PROMPT = """You are a precise, highly analytical Document AI assistant.
Your goal is to answer the user's questions strictly based on the provided document context.
If the answer cannot be found in the context, explicitly state that the document does not contain that information.
Always cite section titles or page numbers when quoting facts."""
