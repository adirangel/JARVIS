"""Utility helpers for dataset collection and preparation.

These functions centralize text sanitation, parsing, and JSONL IO so that
other tooling in :mod:`tools` can share consistent logic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


@dataclass
class TranscriptTurn:
    """Represents a normalized conversational turn."""

    turn_id: str
    conversation_id: str
    speaker: str
    text: str
    source: str
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.turn_id,
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }


def sanitize_text(text: str) -> str:
    """Redact common personally-identifiable information patterns."""

    def _redact(pattern: re.Pattern[str], label: str, value: str) -> str:
        return pattern.sub(f"<{label.upper()}>", value)

    sanitized = text
    sanitized = _redact(EMAIL_RE, "email", sanitized)
    sanitized = _redact(PHONE_RE, "phone", sanitized)
    sanitized = _redact(SSN_RE, "ssn", sanitized)
    sanitized = _redact(CREDIT_CARD_RE, "cc", sanitized)
    return sanitized


def parse_transcript(text: str) -> List[Dict[str, str]]:
    """Parse raw transcript text into a list of ``{"speaker", "text"}`` maps.

    Each line is treated as a single utterance. When the line contains a
    ``Speaker:`` prefix the speaker name is extracted; otherwise ``"unknown"``
    is used.
    """

    turns: List[Dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, utterance = line.split(":", 1)
            turns.append({"speaker": speaker.strip() or "unknown", "text": utterance.strip()})
        else:
            turns.append({"speaker": "unknown", "text": line})
    if not turns:
        turns.append({"speaker": "unknown", "text": ""})
    return turns


def load_metadata(metadata_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not metadata_path:
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Metadata file must contain an object mapping conversation IDs to metadata")
    return {str(k): v for k, v in data.items()}


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")


def iter_transcript_turns(
    path: Path,
    *,
    source: str,
    metadata: Optional[Dict[str, Dict[str, object]]] = None,
    id_prefix: str = "",
) -> Iterator[TranscriptTurn]:
    """Yield sanitized :class:`TranscriptTurn` objects from a transcript file."""

    metadata = metadata or {}
    conversation_id = path.stem
    conversation_meta = metadata.get(conversation_id, {})
    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    for index, turn in enumerate(parse_transcript(raw_text)):
        sanitized_text = sanitize_text(turn["text"])
        turn_id = f"{id_prefix}{conversation_id}-{index:04d}"
        merged_meta = {"source_path": str(path), **conversation_meta}
        yield TranscriptTurn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            speaker=turn["speaker"],
            text=sanitized_text,
            source=source,
            metadata=merged_meta,
        )
