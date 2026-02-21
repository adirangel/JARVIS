"""ChromaDB vector store for semantic memory search. Target: queries <100ms."""

from __future__ import annotations

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

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "jarvis_memory",
        ollama_host: str = "http://localhost:11434",
        cache_recent: bool = True,
        max_cache_size: int = 50,
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

    def search_similar(
        self, query: str, max_results: int = 5, min_score: float = 0.3
    ) -> list[dict]:
        t0 = time.perf_counter()
        if self._collection.count() == 0:
            return []
        # Cache: return cached results for repeated queries (instant, <100ms target)
        cache_key = f"{query[:80]}:{max_results}"
        if self._cache_recent and cache_key in self._cache:
            return self._cache[cache_key]
        query_embedding = self._get_embedding(query)
        n = min(max_results, self._collection.count(), 10)  # Cap for speed
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = 1 - distance
            if score >= min_score:
                matches.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                })
        # Cache recent results (evict oldest if over limit)
        if self._cache_recent and matches:
            if len(self._cache) >= self._cache_max:
                k = next(iter(self._cache))
                del self._cache[k]
            self._cache[cache_key] = matches
        return matches

    def count(self) -> int:
        return self._collection.count()
