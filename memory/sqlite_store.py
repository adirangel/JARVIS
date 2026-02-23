"""SQLite store for facts, conversations, tasks, reminders."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional


class SQLiteStore:
    """Persistent SQLite storage for JARVIS memory."""

    def __init__(self, db_path: str = "data/jarvis.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                due_at REAL,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                trigger_at REAL,
                created_at REAL,
                done INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_at);
        """)
        self._conn.commit()

    def save_message(self, session_id: str, role: str, content: str, timestamp: float) -> None:
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, timestamp),
        )
        self._conn.commit()

    def save_fact(self, key: str, value: str, category: str = "general") -> None:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO facts (key, value, category, updated_at) VALUES (?, ?, ?, ?)",
            (key, value, category, now),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def get_facts(self, category: Optional[str] = None) -> list[dict]:
        if category:
            rows = self._conn.execute(
                "SELECT key, value, category FROM facts WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT key, value, category FROM facts").fetchall()
        return [dict(r) for r in rows]

    def get_recent_conversations(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT session_id, role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit * 2,),  # Get more to have pairs
        ).fetchall()
        return [dict(r) for r in rows]

    def search_conversations(self, query: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT session_id, role, content, timestamp FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 50",
            (f"%{query}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()
        return row["c"] if row else 0

    def add_task(self, description: str, due_at: Optional[float] = None) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO tasks (description, status, due_at, created_at) VALUES (?, 'pending', ?, ?)",
            (description, due_at, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_pending_tasks(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, description, status, due_at FROM tasks WHERE status = 'pending' ORDER BY due_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_reminder(self, text: str, trigger_at: float) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO reminders (text, trigger_at, created_at) VALUES (?, ?, ?)",
            (text, trigger_at, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_reminders(self, before: Optional[float] = None) -> list[dict]:
        now = before or time.time()
        rows = self._conn.execute(
            "SELECT id, text, trigger_at FROM reminders WHERE done = 0 AND trigger_at <= ? ORDER BY trigger_at",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_reminder_done(self, reminder_id: int) -> None:
        self._conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
