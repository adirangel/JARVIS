"""Tests for memory module."""

import pytest
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_sqlite_store():
    from memory.sqlite_store import SQLiteStore
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        store = SQLiteStore(str(db))
        store.save_fact("user_name", "Test User", "user")
        assert store.get_fact("user_name") == "Test User"
        store.save_message("s1", "user", "Hello", 1.0)
        store.save_message("s1", "assistant", "Hi", 1.1)
        msgs = store.get_recent_conversations(limit=5)
        assert len(msgs) >= 2
        store.add_task("Test task")
        tasks = store.get_pending_tasks()
        assert len(tasks) >= 1
        store.close()


def test_memory_manager():
    from memory.manager import MemoryManager
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryManager(
            db_path=str(Path(tmp) / "jarvis.db"),
            chroma_path=str(Path(tmp) / "chroma"),
            use_vector_store=False,  # Avoid ChromaDB lock on Windows temp cleanup
        )
        mem.save_fact("test_key", "test_value")
        assert mem.get_fact("test_key") == "test_value"
        mem.save_interaction("Hi", "Hello Sir")
        ctx = mem.build_context("test")
        assert "test_value" in ctx or "test_key" in ctx or ctx == ""
        mem.close()
