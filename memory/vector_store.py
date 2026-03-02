"""ChromaDB vector store for semantic memory search. Target: queries <100ms."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class VectorStore:
    """Semantic memory using ChromaDB for vector similarity search."""

    # Relevance scoring weights
    _RECENCY_WEIGHT = 0.3        # α — how much recency matters
    _ACCESS_WEIGHT = 0.1         # β — how much recall frequency matters
    _DECAY_LAMBDA = 0.01         # half-life ≈ 70 days

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "jarvis_memory",
        ollama_host: str = "http://localhost:11434",
        cache_recent: bool = True,
        max_cache_size: int = 50,
        sqlite_store=None,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. pip install chromadb")

        base_dir = Path(__file__).parent.parent
        full_path = base_dir / chroma_path
        full_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(full_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model = embedding_model
        self._ollama_host = ollama_host
        self._cache_recent = cache_recent
        self._cache: dict[str, list[dict]] = {}
        self._cache_max = max_cache_size
        self._sqlite = sqlite_store  # for access-count boosting

    def _get_embedding(self, text: str) -> list[float]:
        import ollama
        client = ollama.Client(host=self._ollama_host)
        response = client.embed(model=self._embedding_model, input=text)
        return response["embeddings"][0]

    def store_interaction(
        self,
        interaction_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[dict] = None,
    ) -> None:
        combined = f"Query: {user_message}\nResponse: {assistant_response}"
        meta = {
            "user_message": user_message,
            "assistant_response": assistant_response[:500],
            "timestamp": time.time(),
        }
        if metadata:
            meta.update(metadata)
        embedding = self._get_embedding(combined)
        self._collection.upsert(
            ids=[interaction_id],
            embeddings=[embedding],
            documents=[combined],
            metadatas=[meta],
        )

    def store_summary(self, summary_id: str, summary_text: str, period_start: float, tier: str) -> None:
        """Embed a memory summary so it is searchable."""
        meta = {"type": "summary", "tier": tier, "timestamp": period_start}
        embedding = self._get_embedding(summary_text)
        self._collection.upsert(
            ids=[summary_id],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[meta],
        )

    def _boost_score(self, raw_score: float, memory_id: str, timestamp: float) -> float:
        """Apply recency decay + access-count boost.

        score_final = cosine × (1 + α·recency) × (1 + β·access_count)
        recency = e^(−λ·days_old)
        """
        days_old = max(0, (time.time() - timestamp) / 86400)
        recency = math.exp(-self._DECAY_LAMBDA * days_old)

        access_count = 0
        if self._sqlite:
            try:
                access_count = self._sqlite.get_access_count(memory_id)
            except Exception:
                pass

        boosted = raw_score * (1 + self._RECENCY_WEIGHT * recency) * (1 + self._ACCESS_WEIGHT * access_count)
        return boosted

    def search_similar(
        self, query: str, max_results: int = 5, min_score: float = 0.3
    ) -> list[dict]:
        if self._collection.count() == 0:
            return []
        cache_key = f"{query[:80]}:{max_results}"
        if self._cache_recent and cache_key in self._cache:
            return self._cache[cache_key]
        query_embedding = self._get_embedding(query)
        n = min(max_results * 2, self._collection.count(), 20)  # fetch extra for re-ranking
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else 0
            raw_score = 1 - distance
            if raw_score < min_score:
                continue
            memory_id = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            ts = meta.get("timestamp", time.time())
            boosted = self._boost_score(raw_score, memory_id, ts)
            matches.append({
                "id": memory_id,
                "document": results["documents"][0][i],
                "metadata": meta,
                "score": boosted,
                "raw_score": raw_score,
            })
            # Log access for future boosting
            if self._sqlite:
                try:
                    self._sqlite.log_memory_access(memory_id, query[:200])
                except Exception:
                    pass

        # Re-sort by boosted score, trim to requested count
        matches.sort(key=lambda m: m["score"], reverse=True)
        matches = matches[:max_results]

        if self._cache_recent and matches:
            if len(self._cache) >= self._cache_max:
                k = next(iter(self._cache))
                del self._cache[k]
            self._cache[cache_key] = matches
        return matches

    def count(self) -> int:
        return self._collection.count()
