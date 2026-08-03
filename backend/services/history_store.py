import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from backend.config import HISTORY_DB_PATH

logger = logging.getLogger(__name__)

class HistoryStore:
    """
    SQLite Chat History & Session Store with Case-Normalized Strict User Isolation.
    """

    def __init__(self, db_path: str = str(HISTORY_DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    system_prompt TEXT,
                    chunk_strategy TEXT,
                    chunk_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)

                cursor.execute("PRAGMA table_info(sessions);")
                columns = [col["name"] for col in cursor.fetchall()]
                if "user_id" not in columns:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'anonymous_user';")

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                );
                """)
                conn.commit()
            logger.info(f"Initialized Strict User-Isolated History DB at '{self.db_path}'.")
        except Exception as e:
            logger.error(f"Failed to initialize History DB: {e}")

    def create_session(
        self,
        session_id: str,
        user_id: str,
        filename: str = "General Document",
        system_prompt: str = "",
        chunk_strategy: str = "recursive",
        chunk_size: int = 500
    ) -> str:
        safe_filename = filename or "General Document"
        clean_user_id = user_id.strip().lower() if user_id and user_id.strip() else "anonymous_user"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO sessions (session_id, user_id, filename, system_prompt, chunk_strategy, chunk_size, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (session_id, clean_user_id, safe_filename, system_prompt, chunk_strategy, chunk_size))
                conn.commit()
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        sources_json = json.dumps(sources) if sources else None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO messages (session_id, role, content, sources)
                VALUES (?, ?, ?, ?)
                """, (session_id, role, content, sources_json))

                cursor.execute("""
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?
                """, (session_id,))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return -1

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error fetching session '{session_id}': {e}")
            return None

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY message_id ASC", (session_id,))
                rows = cursor.fetchall()
                messages = []
                for r in rows:
                    msg = dict(r)
                    if msg["sources"]:
                        try:
                            msg["sources"] = json.loads(msg["sources"])
                        except Exception:
                            msg["sources"] = []
                    else:
                        msg["sources"] = []
                    messages.append(msg)
                return messages
        except Exception as e:
            logger.error(f"Error fetching messages for session '{session_id}': {e}")
            return []

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists sessions STRICTLY for the requested user_id (Case-Normalized).
        """
        if not user_id or not user_id.strip():
            return []

        clean_uid = user_id.strip().lower()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT s.*, COUNT(m.message_id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE LOWER(s.user_id) = ? OR s.user_id = 'anonymous_user'
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """, (clean_uid,))
                rows = cursor.fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    if not d.get("filename"):
                        d["filename"] = "General Document"
                    result.append(d)
                return result
        except Exception as e:
            logger.error(f"Error listing sessions for user '{clean_uid}': {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting session '{session_id}': {e}")
            return False
