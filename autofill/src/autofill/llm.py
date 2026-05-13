"""Anthropic SDK wrapper, configured for Computer Use.

Computer Use is a beta — requires the `computer-use-2025-01-24` beta header
and the `computer_20250124` / `bash_20250124` / `text_editor_20250124` tool
versions. The agent here uses the computer tool only — no bash, no text
editor (those would let the model run shell commands which is out of scope
for a browser form-fill task).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    content: list[Any] = field(default_factory=list)  # list of content blocks
    stop_reason: str | None = None
    usage: Any = None
    cost_usd: float = 0.0
    model: str = ""


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
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    display_width: int = 1280,
    display_height: int = 800,
) -> ChatResult:
    """Make a Computer Use chat call.

    Adds the computer tool definition automatically. Uses the Computer
    Use beta header.
    """
    client = _get_client()
    computer_tool = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": display_width,
        "display_height_px": display_height,
        "display_number": 1,
    }
    all_tools = [computer_tool] + (tools or [])
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=all_tools,
        betas=["computer-use-2025-01-24"],
    )
    return ChatResult(
        content=list(response.content),
        stop_reason=response.stop_reason,
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
            "ANTHROPIC_API_KEY is not set. Export it before running a live submission."
        )
