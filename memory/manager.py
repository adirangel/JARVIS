"""Unified memory manager - SQLite + ChromaDB.

Facts, tasks, reminders, conversations, user profile.
Used by agent and heartbeat.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Optional

from memory.sqlite_store import SQLiteStore

try:
    from memory.vector_store import VectorStore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    VectorStore = None


class MemoryManager:
    """Unified memory for JARVIS."""

    def __init__(
        self,
        db_path: str = "data/jarvis.db",
        chroma_path: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        max_memories: int = 5,
        use_vector_store: bool = True,
        chroma_cache_recent: bool = True,
        chroma_cache_size: int = 50,
    ):
        self._sqlite = SQLiteStore(db_path)
        self._max_memories = max_memories
        self._session_id = str(uuid.uuid4())
        self._vector_store: Optional[VectorStore] = None
        if use_vector_store and CHROMADB_AVAILABLE and VectorStore:
            try:
                self._vector_store = VectorStore(
                    chroma_path=chroma_path,
                    embedding_model=embedding_model,
                    ollama_host=ollama_host,
                    cache_recent=chroma_cache_recent,
                    max_cache_size=chroma_cache_size,
                )
            except Exception:
                self._vector_store = None

        self._fact_patterns = [
            (r"(?:קוראים לי|שמי|אני)\s+(.+?)(?:\s*[.,!]|$)", "user_name", "user"),
            (r"(?:אני גר ב|אני מ)(.+?)(?:\s*[.,!]|$)", "user_location", "user"),
            (r"(?:אני עובד ב|אני עובד כ)(.+?)(?:\s*[.,!]|$)", "user_work", "user"),
            (r"(?:אני אוהב|אני מעדיף)\s+(.+?)(?:\s*[.,!]|$)", "user_preference", "preferences"),
            (r"(?:יום ההולדת שלי|נולדתי ב)\s*(.+?)(?:\s*[.,!]|$)", "user_birthday", "user"),
        ]

    @property
    def session_id(self) -> str:
        return self._session_id

    def save_interaction(self, user_message: str, assistant_response: str) -> None:
        ts = time.time()
        self._sqlite.save_message(self._session_id, "user", user_message, ts)
        self._sqlite.save_message(self._session_id, "assistant", assistant_response, ts + 0.001)
        if self._vector_store:
            try:
                self._vector_store.store_interaction(
                    f"{self._session_id}_{int(ts)}",
                    user_message,
                    assistant_response,
                )
            except Exception:
                pass
        self._extract_facts(user_message)

    def _extract_facts(self, text: str) -> None:
        for pattern, key, category in self._fact_patterns:
            m = re.search(pattern, text)
            if m:
                val = m.group(1).strip()
                if val and len(val) > 1:
                    self.save_fact(key, val, category)

    def save_fact(self, key: str, value: str, category: str = "general") -> None:
        self._sqlite.save_fact(key, value, category)

    def get_fact(self, key: str) -> Optional[str]:
        return self._sqlite.get_fact(key)

    def get_all_facts(self, category: Optional[str] = None) -> list[dict]:
        return self._sqlite.get_facts(category)

    def build_context(self, query: str) -> str:
        parts = []
        facts = self._sqlite.get_facts()
        if facts:
            parts.append("Known facts:\n" + "\n".join(f"  - {f['key']}: {f['value']}" for f in facts))
        if self._vector_store and self._vector_store.count() > 0:
            try:
                similar = self._vector_store.search_similar(query, max_results=self._max_memories)
                if similar:
                    lines = [
                        f"  - [Relevance: {s['score']:.0%}] {s['metadata'].get('user_message', '?')} | {str(s['metadata'].get('assistant_response', ''))[:200]}"
                        for s in similar
                    ]
                    parts.append("Relevant past context:\n" + "\n".join(lines))
            except Exception:
                pass
        return "\n\n".join(parts) if parts else ""

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        return self._sqlite.get_recent_conversations(limit=limit)

    def get_pending_tasks(self) -> list[dict]:
        """For heartbeat - tasks to execute."""
        return self._sqlite.get_pending_tasks()

    def get_reminders(self) -> list[dict]:
        """For heartbeat - reminders due now."""
        return self._sqlite.get_reminders()

    def add_task(self, description: str, due_at: Optional[float] = None) -> int:
        return self._sqlite.add_task(description, due_at)

    def add_reminder(self, text: str, trigger_at: float) -> int:
        return self._sqlite.add_reminder(text, trigger_at)

    def mark_reminder_done(self, reminder_id: int) -> None:
        self._sqlite.mark_reminder_done(reminder_id)

    def close(self) -> None:
        self._sqlite.close()
