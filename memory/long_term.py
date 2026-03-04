"""Long-term semantic memory.  Facts + vector search + user profile + consolidation."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory.manager import MemoryManager


class LongTermMemory:
    """Persistent memory: facts (SQLite) + semantic search (ChromaDB) + user profile."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_fn: Optional[Callable] = None, embed_fn=None):
        config = config or {}
        mem_cfg = config.get("memory", config)
        self._manager = MemoryManager(
            db_path=mem_cfg.get("db_path", "data/jarvis.db"),
            chroma_path=mem_cfg.get("chroma_path", "data/chroma"),
            embedding_model=mem_cfg.get("embedding_model", "nomic-embed-text"),
            ollama_host=mem_cfg.get("ollama_host", "http://localhost:11434"),
            max_memories=mem_cfg.get("max_memories", 5),
            chroma_cache_recent=mem_cfg.get("chroma_cache_recent", True),
            chroma_cache_size=mem_cfg.get("chroma_cache_size", 50),
            use_vector_store=True,
            llm_fn=llm_fn,
            embed_fn=embed_fn,
        )

    @property
    def manager(self) -> MemoryManager:
        return self._manager

    def set_llm_fn(self, fn: Callable[[str], str]) -> None:
        self._manager.set_llm_fn(fn)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return semantically similar past context as (doc, score, metadata)."""
        results = self._manager.retrieve_similar(query, max_results=k)
        return [
            (r["document"], r["score"], r.get("metadata", {}))
            for r in results
        ]

    def build_context(self, query: str, token_budget: int = 3000) -> str:
        """Assemble layered memory context for the LLM prompt."""
        return self._manager.build_context(query, token_budget=token_budget)

    def get_profile_prompt(self) -> str:
        """Return the user profile block for system prompt injection."""
        return self._manager.get_profile_prompt()

    # ── Storage ───────────────────────────────────────────────────────────────

    def save_interaction(self, user_message: str, assistant_response: str) -> None:
        """Persist an exchange (SQLite + ChromaDB + fact extraction)."""
        self._manager.save_interaction(user_message, assistant_response)

    def store(self, text: str, category: str = "general") -> None:
        """Store a standalone fact."""
        key = f"fact_{category}_{int(time.time() * 1000)}"
        self._manager.save_fact(key, text, category)

    # ── Consolidation (called by heartbeat) ───────────────────────────────────

    def consolidate(self, age_hours: float = 24.0) -> int:
        return self._manager.consolidate(age_hours=age_hours)

    def consolidate_weekly(self) -> Optional[str]:
        return self._manager.consolidate_weekly()
