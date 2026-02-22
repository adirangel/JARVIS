"""Latency benchmarking and context token tracking.

When config.timing=true: log timings for each major step.
When config.context.show_after_each_turn: track and display token usage.
"""

from __future__ import annotations

import contextvars
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object  # type: ignore

# Session token accumulator (prompt + completion from every LLM call)
_session_tokens: contextvars.ContextVar[int] = contextvars.ContextVar("session_tokens", default=0)

# Timing log file path
_TIMING_LOG = Path("logs/timing.log")


def reset_session_tokens() -> None:
    """Reset token count for new session."""
    _session_tokens.set(0)


def add_tokens(prompt_count: int, completion_count: int) -> None:
    """Add tokens from an LLM call to session total."""
    cur = _session_tokens.get()
    _session_tokens.set(cur + prompt_count + completion_count)


def get_session_tokens() -> int:
    """Get cumulative tokens used in current session."""
    return _session_tokens.get()


def format_context_display(used: int, max_tokens: int, pct: float) -> str:
    """Format context usage for display/speech."""
    return f"[Context: {used:,} / {max_tokens:,} tokens ({pct:.0f}%)]"


def get_context_status(config: Optional[dict] = None) -> tuple[str, bool]:
    """Return (display_string, is_warning). Warning when above threshold."""
    cfg = config or {}
    ctx_cfg = cfg.get("context", {})
    max_tok = int(ctx_cfg.get("max_tokens", 256,000))
    threshold = float(ctx_cfg.get("warning_threshold", 0.85))
    used = get_session_tokens()
    pct = (used / max_tok) if max_tok > 0 else 0.0
    return format_context_display(used, max_tok, pct * 100), pct >= threshold


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback to capture Ollama prompt_eval_count and eval_count."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self._config = config or {}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extract token counts from LLMResult or AIMessage."""
        prompt_count = 0
        eval_count = 0
        # LLMResult has generations; each may have message.response_metadata
        if hasattr(response, "generations"):
            for gen_list in (response.generations or []):
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "response_metadata", None) or {}
                    prompt_count += int(meta.get("prompt_eval_count", 0))
                    eval_count += int(meta.get("eval_count", 0))
        # Direct AIMessage (e.g. from streaming)
        else:
            meta = getattr(response, "response_metadata", None) or {}
            prompt_count = int(meta.get("prompt_eval_count", 0))
            eval_count = int(meta.get("eval_count", 0))
        if prompt_count or eval_count:
            add_tokens(prompt_count, eval_count)


def log_timing(
    step: str,
    elapsed_ms: float,
    config: Optional[dict] = None,
    verbose: bool = True,
) -> None:
    """Log timing to console and optionally to logs/timing.log."""
    cfg = config or {}
    if not cfg.get("timing", False) and not cfg.get("debug", False):
        return
    msg = f"{step}: {elapsed_ms:.0f}ms"
    if verbose:
        print(f"[Timing] {msg}", flush=True)
    log_path = cfg.get("timing_log") or str(_TIMING_LOG)
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except Exception as e:
            logger.debug("Could not write timing log: %s", e)
