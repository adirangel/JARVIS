"""Unified memory manager — SQLite + ChromaDB.

Provides:
  - save_interaction()       — persist every exchange
  - extract_facts_from()     — 2-stage LLM fact extraction
  - consolidate()            — daily / weekly / monthly summaries
  - build_context()          — layered context assembly for LLM prompt
  - User profile CRUD
  - Facts, tasks, reminders
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from memory.sqlite_store import SQLiteStore

try:
    from memory.vector_store import VectorStore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    VectorStore = None


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return max(1, len(text) // 4)


# ── LLM prompts for memory extraction ────────────────────────────────────────

_GATE_PROMPT = (
    "Does the following user message contain any personal information about the "
    "user?  Personal information includes: their name, age, location, job, family, "
    "preferences, routines, schedule, relationships, opinions, hobbies, or anything "
    "factual about them.\n\n"
    "Message: \"{message}\"\n\n"
    "Reply with ONLY the word YES or NO."
)

_EXTRACT_PROMPT = (
    "You are a memory extraction system.  Given the user message and the "
    "assistant response, extract structured facts about the user.\n\n"
    "User: {user_message}\n"
    "Assistant: {assistant_response}\n\n"
    "Return a JSON array of objects.  Each object has:\n"
    "  - \"key\": a dot-path like \"name\", \"preferences.coffee\", "
    "\"schedule.morning\", \"relationships.wife.name\"\n"
    "  - \"value\": the extracted fact (short string)\n"
    "  - \"confidence\": 0.0-1.0 (how certain this is)\n\n"
    "Only include things explicitly stated or strongly implied.  "
    "Return an empty array [] if nothing new.\n"
    "Return ONLY valid JSON, no markdown fences."
)

_CONSOLIDATION_PROMPT = (
    "You are a memory consolidation system for a personal AI assistant named JARVIS.\n"
    "Below are conversation messages from {date_label}.\n"
    "Extract:\n"
    "1. A concise 2-3 sentence summary of what was discussed.\n"
    "2. Key facts learned about the user (JSON array).\n"
    "3. People, places, projects mentioned (JSON array of strings).\n\n"
    "Messages:\n{messages_text}\n\n"
    "Reply with ONLY valid JSON (no markdown fences):\n"
    '{{"summary": "...", "facts": [{{"key": "...", "value": "...", "confidence": 0.9}}], '
    '"entities": ["..."]}}'
)


class MemoryManager:
    """Unified memory for JARVIS — forever recall, nothing forgotten."""

    def __init__(
        self,
        db_path: str = "data/jarvis.db",
        chroma_path: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        max_memories: int = 5,
        chroma_cache_recent: bool = True,
        chroma_cache_size: int = 50,
        use_vector_store: bool = True,
        vector_store=None,
        embed_fn=None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self._session_id = str(uuid.uuid4())
        self._max_memories = max_memories
        self._sqlite = SQLiteStore(db_path)
        self._turn_counter = 0
        self._llm_fn = llm_fn          # sync function: prompt → response text

        self.use_vector_store = use_vector_store and CHROMADB_AVAILABLE
        self._vector_store: Optional[VectorStore] = None
        if self.use_vector_store:
            if vector_store:
                self._vector_store = vector_store
            else:
                try:
                    self._vector_store = VectorStore(
                        chroma_path=chroma_path,
                        embedding_model=embedding_model,
                        ollama_host=ollama_host,
                        cache_recent=chroma_cache_recent,
                        max_cache_size=chroma_cache_size,
                        sqlite_store=self._sqlite,
                        embed_fn=embed_fn,
                    )
                    logger.info("VectorStore initialised")
                except Exception as e:
                    logger.error(f"VectorStore init failed: {e}")
                    self._vector_store = None
                    self.use_vector_store = False

    # ── Public properties ─────────────────────────────────────────────────────

    def session_id(self) -> str:
        return self._session_id

    def set_llm_fn(self, fn: Callable[[str], str]) -> None:
        """Set (or update) the LLM function used for extraction / consolidation."""
        self._llm_fn = fn

    # ── Save interaction ──────────────────────────────────────────────────────

    def save_interaction(self, user_message: str, assistant_response: str) -> None:
        """Persist an exchange to SQLite + ChromaDB, then run fact extraction."""
        ts = time.time()
        self._sqlite.save_message(self._session_id, "user", user_message, ts)
        self._sqlite.save_message(self._session_id, "assistant", assistant_response, ts + 0.001)

        # Embed in vector store
        if self._vector_store:
            try:
                self._vector_store.store_interaction(
                    f"{self._session_id}_{int(ts)}",
                    user_message,
                    assistant_response,
                )
            except Exception as e:
                logger.debug(f"Vector store error: {e}")

        # Every 3rd turn, attempt LLM fact extraction (async-safe: runs sync here)
        self._turn_counter += 1
        if self._turn_counter % 3 == 0:
            self.extract_facts_from(user_message, assistant_response)

    # ── 2-stage LLM fact extraction ───────────────────────────────────────────

    def extract_facts_from(self, user_message: str, assistant_response: str) -> list[dict]:
        """2-stage gate: cheap YES/NO → full extraction.  Returns extracted facts."""
        if not self._llm_fn:
            return self._extract_facts_regex(user_message)

        try:
            # Stage 1: cheap gate
            gate_prompt = _GATE_PROMPT.format(message=user_message[:500])
            gate_answer = self._llm_fn(gate_prompt).strip().upper()
            if "YES" not in gate_answer:
                return []

            # Stage 2: full extraction
            prompt = _EXTRACT_PROMPT.format(
                user_message=user_message[:800],
                assistant_response=assistant_response[:800],
            )
            raw = self._llm_fn(prompt).strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            facts = json.loads(raw)
            if not isinstance(facts, list):
                return []
            for fact in facts:
                key = fact.get("key", "")
                value = fact.get("value", "")
                conf = float(fact.get("confidence", 0.7))
                if key and value:
                    self._sqlite.upsert_profile(key, value, conf)
                    logger.debug(f"[Memory] Extracted: {key} = {value} ({conf:.0%})")
            return facts
        except Exception as e:
            logger.debug(f"[Memory] Fact extraction error: {e}")
            return []

    def _extract_facts_regex(self, text: str) -> list[dict]:
        """Fallback regex extraction when no LLM is available."""
        facts = []
        lower = text.lower()
        # "my name is X"
        m = re.search(r"my name is (\w+)", lower)
        if m:
            name = m.group(1).title()
            self._sqlite.upsert_profile("name", name, 0.95)
            facts.append({"key": "name", "value": name, "confidence": 0.95})
        # Preferences
        for trigger in ("i prefer", "i like", "i love", "i hate", "i enjoy"):
            if trigger in lower:
                idx = lower.index(trigger) + len(trigger)
                pref = text[idx:idx + 80].strip(" ,.!?")
                if pref:
                    key = f"preferences.{trigger.split()[-1]}"
                    self._sqlite.upsert_profile(key, pref, 0.7)
                    facts.append({"key": key, "value": pref, "confidence": 0.7})
        return facts

    # ── Facts ─────────────────────────────────────────────────────────────────

    def save_fact(self, key: str, value: str, category: str = "general") -> None:
        self._sqlite.save_fact(key, value, category)

    def get_fact(self, key: str) -> Optional[str]:
        return self._sqlite.get_fact(key)

    def get_all_facts(self, category: Optional[str] = None) -> list[dict]:
        return self._sqlite.get_facts(category)

    # ── User profile ──────────────────────────────────────────────────────────

    def upsert_profile(self, key: str, value: str, confidence: float = 0.8) -> None:
        self._sqlite.upsert_profile(key, value, confidence)

    def get_user_profile(self) -> list[dict]:
        return self._sqlite.get_user_profile()

    def get_profile_prompt(self) -> str:
        return self._sqlite.format_profile_for_prompt()

    # ── Semantic search ───────────────────────────────────────────────────────

    def retrieve_similar(self, query: str, max_results: int = 5) -> list[dict]:
        if not self._vector_store or self._vector_store.count() == 0:
            return []
        try:
            return self._vector_store.search_similar(query, max_results=max_results)
        except Exception:
            return []

    # ── Consolidation (daily / weekly / monthly) ──────────────────────────────

    def consolidate(self, age_hours: float = 24.0) -> int:
        """Summarise messages older than *age_hours* into daily summaries.

        Returns the number of new summaries created.
        """
        if not self._llm_fn:
            logger.debug("[Memory] No LLM function — skipping consolidation")
            return 0

        cutoff = time.time() - age_hours * 3600
        messages = self._sqlite.get_unconsolidated_messages(cutoff)
        if not messages:
            return 0

        # Group by calendar day
        days: dict[str, list[dict]] = {}
        for msg in messages:
            day_key = datetime.fromtimestamp(msg["timestamp"]).strftime("%Y-%m-%d")
            days.setdefault(day_key, []).append(msg)

        created = 0
        for day_key, day_msgs in days.items():
            if len(day_msgs) < 2:      # skip near-empty days
                continue
            messages_text = "\n".join(
                f"[{m['role']}] {m['content'][:300]}" for m in day_msgs
            )
            prompt = _CONSOLIDATION_PROMPT.format(
                date_label=day_key,
                messages_text=messages_text[:4000],
            )
            try:
                raw = self._llm_fn(prompt).strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
            except Exception as e:
                logger.warning(f"[Memory] Consolidation parse error for {day_key}: {e}")
                continue

            summary_text = data.get("summary", "")
            facts = data.get("facts", [])
            entities = data.get("entities", [])
            if not summary_text:
                continue

            period_start = min(m["timestamp"] for m in day_msgs)
            period_end = max(m["timestamp"] for m in day_msgs)

            sid = self._sqlite.save_summary(
                period_start=period_start,
                period_end=period_end,
                tier="daily",
                summary=summary_text,
                facts_extracted=facts,
                entities=entities,
                message_count=len(day_msgs),
            )

            # Embed summary for semantic search
            if self._vector_store:
                try:
                    self._vector_store.store_summary(
                        f"summary_daily_{day_key}",
                        summary_text,
                        period_start,
                        "daily",
                    )
                except Exception:
                    pass

            # Upsert extracted facts into user profile
            for fact in facts:
                key = fact.get("key", "")
                value = fact.get("value", "")
                conf = float(fact.get("confidence", 0.7))
                if key and value:
                    self._sqlite.upsert_profile(key, value, conf)

            # Mark messages as consolidated
            ids = [m["id"] for m in day_msgs]
            self._sqlite.mark_messages_consolidated(ids)
            created += 1
            logger.info(f"[Memory] Consolidated {len(day_msgs)} messages for {day_key}")

        return created

    def consolidate_weekly(self) -> Optional[str]:
        """Create a weekly summary from the last 7 days of daily summaries."""
        if not self._llm_fn:
            return None
        week_ago = time.time() - 7 * 86400
        dailies = self._sqlite.get_summaries("daily", since=week_ago)
        if len(dailies) < 2:
            return None
        combined = "\n\n".join(
            f"[{datetime.fromtimestamp(d['period_start']).strftime('%A %b %d')}] {d['summary']}"
            for d in dailies
        )
        prompt = (
            "Combine these daily summaries into one concise weekly summary (3-5 sentences).\n"
            "Focus on recurring themes, important facts, and the user's priorities.\n\n"
            f"{combined[:4000]}\n\nWeekly summary:"
        )
        try:
            summary = self._llm_fn(prompt).strip()
            period_start = min(d["period_start"] for d in dailies)
            period_end = max(d["period_end"] for d in dailies)
            self._sqlite.save_summary(period_start, period_end, "weekly", summary,
                                      message_count=sum(d["message_count"] for d in dailies))
            if self._vector_store:
                self._vector_store.store_summary(
                    f"summary_weekly_{int(period_start)}", summary, period_start, "weekly"
                )
            logger.info("[Memory] Weekly summary created")
            return summary
        except Exception as e:
            logger.warning(f"[Memory] Weekly consolidation error: {e}")
            return None

    # ── Layered context assembly ──────────────────────────────────────────────

    def build_context(self, query: str, token_budget: int = 3000) -> str:
        """Assemble memory context for LLM, within a token budget.

        Layers (highest priority first):
          1. User profile   (~300-500 tokens, always)
          2. Known facts     (~200 tokens)
          3. Today's summary (~200 tokens)
          4. Semantic search (fills remaining budget)
        """
        sections: list[str] = []
        remaining = token_budget

        # 1. User profile (always injected)
        profile_text = self._sqlite.format_profile_for_prompt()
        if profile_text:
            sections.append(profile_text)
            remaining -= _estimate_tokens(profile_text)

        # 2. Known facts
        facts = self._sqlite.get_facts()
        if facts and remaining > 100:
            facts_text = "\n".join(f"- {f['key']}: {f['value']}" for f in facts[:15])
            block = f"Known facts:\n{facts_text}"
            cost = _estimate_tokens(block)
            if cost <= remaining:
                sections.append(block)
                remaining -= cost

        # 3. Today's daily summary
        today_summary = self._sqlite.get_latest_summary("daily")
        if today_summary and remaining > 100:
            ts = today_summary["period_start"]
            day_label = datetime.fromtimestamp(ts).strftime("%A %b %d")
            block = f"Earlier ({day_label}):\n{today_summary['summary']}"
            cost = _estimate_tokens(block)
            if cost <= remaining:
                sections.append(block)
                remaining -= cost

        # 4. Semantic search (remaining budget)
        if self._vector_store and remaining > 200:
            max_results = max(1, min(5, remaining // 350))
            try:
                similar = self._vector_store.search_similar(query, max_results=max_results)
                if similar:
                    lines = []
                    for s in similar:
                        user_q = s["metadata"].get("user_message", "")
                        asst_a = str(s["metadata"].get("assistant_response", ""))[:200]
                        score = s.get("score", 0)
                        line = f"- [{score:.0%}] Q: {user_q[:120]} → A: {asst_a}"
                        cost = _estimate_tokens(line)
                        if cost > remaining:
                            break
                        lines.append(line)
                        remaining -= cost
                    if lines:
                        sections.append("Relevant past conversations:\n" + "\n".join(lines))
            except Exception:
                pass

        return "\n\n".join(sections) if sections else ""

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        return self._sqlite.get_recent_conversations(limit=limit)

    def get_pending_tasks(self) -> list[dict]:
        return self._sqlite.get_pending_tasks()

    def get_reminders(self) -> list[dict]:
        return self._sqlite.get_reminders()

    def add_task(self, description: str, due_at: Optional[float] = None) -> int:
        return self._sqlite.add_task(description, due_at)

    def add_reminder(self, text: str, trigger_at: float) -> int:
        return self._sqlite.add_reminder(text, trigger_at)

    def mark_reminder_done(self, reminder_id: int) -> None:
        self._sqlite.mark_reminder_done(reminder_id)

    def close(self) -> None:
        self._sqlite.close()
