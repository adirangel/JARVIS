"""SQLite store for facts, conversations, tasks, reminders, user profile, memory summaries."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional


class SQLiteStore:
    """Persistent SQLite storage for JARVIS memory."""

    def __init__(self, db_path: str = "data/jarvis.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

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
                timestamp REAL,
                consolidated INTEGER DEFAULT 0
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
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                first_mentioned REAL,
                last_mentioned REAL,
                mention_count INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS memory_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                tier TEXT NOT NULL,
                summary TEXT NOT NULL,
                facts_extracted TEXT DEFAULT '[]',
                entities TEXT DEFAULT '[]',
                message_count INTEGER DEFAULT 0,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                accessed_at REAL,
                query_context TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_messages_consolidated ON messages(consolidated);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders(trigger_at);
            CREATE INDEX IF NOT EXISTS idx_summaries_tier ON memory_summaries(tier, period_start);
            CREATE INDEX IF NOT EXISTS idx_access_log_memory ON memory_access_log(memory_id);

            CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                task_type TEXT DEFAULT 'general',
                task_params TEXT DEFAULT '{}',
                status TEXT DEFAULT 'idle',
                result TEXT DEFAULT '',
                error TEXT DEFAULT '',
                created_at REAL,
                started_at REAL,
                finished_at REAL
            );
        """)
        # Migrate: add 'consolidated' column if missing (existing DBs)
        try:
            self._conn.execute("SELECT consolidated FROM messages LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE messages ADD COLUMN consolidated INTEGER DEFAULT 0")
            self._conn.commit()
        self._conn.commit()

    # ── Messages ──────────────────────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str, timestamp: float) -> None:
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, timestamp),
        )
        self._conn.commit()

    def get_recent_conversations(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT session_id, role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit * 2,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_conversations(self, query: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT session_id, role, content, timestamp FROM messages "
            "WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 50",
            (f"%{query}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()
        return row["c"] if row else 0

    def get_unconsolidated_messages(self, before_ts: float) -> list[dict]:
        """Return messages not yet summarised, older than *before_ts*."""
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, timestamp FROM messages "
            "WHERE consolidated = 0 AND timestamp < ? ORDER BY timestamp",
            (before_ts,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_messages_consolidated(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(
            f"UPDATE messages SET consolidated = 1 WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()

    # ── Facts ─────────────────────────────────────────────────────────────────

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

    # ── User profile ──────────────────────────────────────────────────────────

    def upsert_profile(self, key: str, value: str, confidence: float = 0.8) -> None:
        """Insert or update a user-profile entry.  Repeated mentions boost confidence."""
        now = time.time()
        existing = self._conn.execute(
            "SELECT mention_count, confidence FROM user_profile WHERE key = ?", (key,)
        ).fetchone()
        if existing:
            new_count = existing["mention_count"] + 1
            # Confidence converges toward 1.0 with more mentions
            new_conf = min(1.0, existing["confidence"] + (1.0 - existing["confidence"]) * 0.15)
            self._conn.execute(
                "UPDATE user_profile SET value=?, confidence=?, last_mentioned=?, mention_count=? WHERE key=?",
                (value, new_conf, now, new_count, key),
            )
        else:
            self._conn.execute(
                "INSERT INTO user_profile (key, value, confidence, first_mentioned, last_mentioned, mention_count) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (key, value, confidence, now, now),
            )
        self._conn.commit()

    def get_user_profile(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT key, value, confidence, mention_count FROM user_profile ORDER BY confidence DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_profile_value(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM user_profile WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def format_profile_for_prompt(self) -> str:
        """Return a compact text block of the user profile for system prompt injection."""
        rows = self.get_user_profile()
        if not rows:
            return ""
        lines = []
        for r in rows:
            conf = r["confidence"]
            tag = "" if conf >= 0.7 else " (uncertain)"
            lines.append(f"- {r['key']}: {r['value']}{tag}")
        return "What you know about the user:\n" + "\n".join(lines)

    # ── Memory summaries ──────────────────────────────────────────────────────

    def save_summary(
        self,
        period_start: float,
        period_end: float,
        tier: str,
        summary: str,
        facts_extracted: list | None = None,
        entities: list | None = None,
        message_count: int = 0,
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO memory_summaries "
            "(period_start, period_end, tier, summary, facts_extracted, entities, message_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                period_start,
                period_end,
                tier,
                summary,
                json.dumps(facts_extracted or []),
                json.dumps(entities or []),
                message_count,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_summaries(self, tier: str, since: Optional[float] = None) -> list[dict]:
        if since:
            rows = self._conn.execute(
                "SELECT * FROM memory_summaries WHERE tier = ? AND period_start >= ? ORDER BY period_start",
                (tier, since),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_summaries WHERE tier = ? ORDER BY period_start DESC LIMIT 30",
                (tier,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["facts_extracted"] = json.loads(d.get("facts_extracted") or "[]")
            d["entities"] = json.loads(d.get("entities") or "[]")
            result.append(d)
        return result

    def get_latest_summary(self, tier: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM memory_summaries WHERE tier = ? ORDER BY period_end DESC LIMIT 1",
            (tier,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["facts_extracted"] = json.loads(d.get("facts_extracted") or "[]")
        d["entities"] = json.loads(d.get("entities") or "[]")
        return d

    # ── Memory access log ─────────────────────────────────────────────────────

    def log_memory_access(self, memory_id: str, query_context: str = "") -> None:
        self._conn.execute(
            "INSERT INTO memory_access_log (memory_id, accessed_at, query_context) VALUES (?, ?, ?)",
            (memory_id, time.time(), query_context),
        )
        self._conn.commit()

    def get_access_count(self, memory_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM memory_access_log WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return row["c"] if row else 0

    # ── Tasks ─────────────────────────────────────────────────────────────────

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

    # ── Reminders ─────────────────────────────────────────────────────────────

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

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
