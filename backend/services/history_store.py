import sqlite3
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional
from backend.config import HISTORY_DB_PATH

class HistoryStore:
    """
    SQLite-backed Chat History & Session persistence service.
    Saved directly on Railway volume storage for persistent recall.
    """

    def __init__(self):
        self.db_path = str(HISTORY_DB_PATH)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    chunk_strategy TEXT,
                    chunk_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def create_session(
        self,
        filename: str,
        system_prompt: str,
        chunk_strategy: str = "recursive",
        chunk_size: int = 500
    ) -> str:
        session_id = str(uuid.uuid4())[:8]
        now = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, filename, system_prompt, chunk_strategy, chunk_size, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, filename, system_prompt, chunk_strategy, chunk_size, now, now))
            conn.commit()
        return session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.session_id, s.filename, s.system_prompt, s.chunk_strategy, s.created_at, s.updated_at,
                       COUNT(m.message_id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: List[Dict[str, Any]] = None
    ) -> str:
        message_id = str(uuid.uuid4())[:8]
        now = datetime.datetime.utcnow().isoformat()
        sources_str = json.dumps(sources) if sources else "[]"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, sources_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (message_id, session_id, role, content, sources_str, now))
            
            cursor.execute("""
                UPDATE sessions SET updated_at = ? WHERE session_id = ?
            """, (now, session_id))
            conn.commit()

        return message_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, role, content, sources_json, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            rows = cursor.fetchall()
            
            messages = []
            for r in rows:
                m = dict(r)
                try:
                    m["sources"] = json.loads(m.get("sources_json", "[]"))
                except Exception:
                    m["sources"] = []
                messages.append(m)
            return messages

    def delete_session(self, session_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
