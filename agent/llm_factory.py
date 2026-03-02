"""LLM factory: Ollama (local) or OpenRouter (API)."""

from __future__ import annotations

import os
from typing import Any, Optional


def get_llm(
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
    provider: str = "ollama",
    api_key: Optional[str] = None,
) -> Any:
    """Create LLM: ChatOpenAI for OpenRouter, ChatOllama for local Ollama."""
    if provider == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenRouter requires an API key. Set llm.api_key in config.yaml or OPENROUTER_API_KEY env var."
            )
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": model,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": key,
            "temperature": temperature,
        }
        if num_predict is not None:
            kwargs["max_tokens"] = num_predict
        return ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama

    kwargs = {"model": model, "base_url": base_url, "temperature": temperature}
    model_kwargs: dict = {}
    if num_predict is not None:
        model_kwargs["num_predict"] = num_predict
    if num_ctx is not None:
        model_kwargs["num_ctx"] = num_ctx
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    return ChatOllama(**kwargs)


def get_llm_from_config(
    llm_cfg: dict,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> Any:
    """Build LLM from config. Supports provider=openrouter with api_key."""
    llm_cfg = llm_cfg or {}
    provider = llm_cfg.get("provider", "ollama")
    model = (
        llm_cfg.get("model")
        or llm_cfg.get("conversation_model")
        or llm_cfg.get("tool_model", "qwen3:4b")
    )
    base_url = llm_cfg.get("host", "http://localhost:11434")
    if base_url and not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY") or ""

    return get_llm(
        model=model,
        base_url=base_url,
        temperature=temperature or llm_cfg.get("temperature", 0.7),
        num_predict=num_predict or llm_cfg.get("max_tokens"),
        num_ctx=num_ctx,
        provider=provider,
        api_key=api_key or None,
    )
