"""LLM client — thin wrapper over the Anthropic SDK with prompt caching.

Every agent's persona + the scenario setting + the shared goal are stable
across all ticks for that agent — perfect prompt-caching target. Each
agent gets its own conversation thread (its own `messages` history), so
its system prompt (persona + setting + goal) is set once and reused.

This module exposes a `_chat_fn` hook so tests can substitute a mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic


_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 5e-6, "output": 25e-6},
    "claude-sonnet-4-6": {"input": 3e-6, "output": 15e-6},
    "claude-haiku-4-5": {"input": 1e-6, "output": 5e-6},
}


def estimate_cost(model: str, usage: anthropic.types.Usage) -> float:
    price = _PRICES.get(model, _PRICES["claude-sonnet-4-6"])
    inp = price["input"]
    out = price["output"]
    cache_write = inp * 1.25
    cache_read = inp * 0.1
    return (
        (usage.input_tokens or 0) * inp
        + (usage.output_tokens or 0) * out
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * cache_write
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cache_read
    )


@dataclass
class ChatResult:
    text: str
    usage: anthropic.types.Usage
    cost_usd: float
    model: str


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _real_chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
) -> ChatResult:
    client = _get_client()
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=messages,
    )
    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    return ChatResult(
        text=text.strip(),
        usage=response.usage,
        cost_usd=estimate_cost(model, response.usage),
        model=response.model,
    )


_chat_fn: Callable[..., ChatResult] = _real_chat


def chat(**kwargs: Any) -> ChatResult:
    return _chat_fn(**kwargs)


def set_chat_fn(fn: Callable[..., ChatResult]) -> None:
    global _chat_fn
    _chat_fn = fn


def reset_chat_fn() -> None:
    global _chat_fn
    _chat_fn = _real_chat


def require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running a scenario:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
