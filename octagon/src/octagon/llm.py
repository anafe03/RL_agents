"""LLM client — thin wrapper over the Anthropic SDK with prompt caching and cost tracking.

The defender's system prompt and tool schemas are identical across every attack
in an audit, so they're an ideal prompt-caching target. We put a
`cache_control` breakpoint on the last system block — that caches tools +
system together (tools render before system), giving us ~90% cost reduction
on input tokens after the first attack.

This module also exposes a `_chat_fn` hook so tests can substitute a mock
without monkey-patching `anthropic.Anthropic`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic

# --- pricing ----------------------------------------------------------------
# $ per token. Cache writes are 1.25x base input; cache reads are 0.1x base.
# Source: shared/models.md (cached 2026-04-29).
_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 5e-6, "output": 25e-6},
    "claude-opus-4-6": {"input": 5e-6, "output": 25e-6},
    "claude-sonnet-4-6": {"input": 3e-6, "output": 15e-6},
    "claude-haiku-4-5": {"input": 1e-6, "output": 5e-6},
}


def estimate_cost(model: str, usage: anthropic.types.Usage) -> float:
    """Compute USD cost from an Anthropic Usage object, accounting for caching."""
    price = _PRICES.get(model, _PRICES["claude-sonnet-4-6"])  # safe default
    inp = price["input"]
    out = price["output"]
    cache_write = inp * 1.25
    cache_read = inp * 0.1
    cost = (
        (usage.input_tokens or 0) * inp
        + (usage.output_tokens or 0) * out
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * cache_write
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cache_read
    )
    return cost


# --- chat call --------------------------------------------------------------


@dataclass
class ChatResult:
    """Normalized response from a single chat call."""

    content: list[Any]  # list of content blocks (text, tool_use, etc.)
    stop_reason: str | None
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
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    thinking_mode: str = "disabled",  # "disabled" | "adaptive"
    effort: str | None = None,        # only used when thinking is adaptive
) -> ChatResult:
    """Make one chat call to Anthropic.

    System prompt and tools share one cache_control breakpoint on the last
    system block — tools render before system, so this caches both.
    """
    client = _get_client()

    # Build system as a list with a cache breakpoint on the last block.
    system_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if thinking_mode == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
        if effort:
            kwargs["output_config"] = {"effort": effort}

    response = client.messages.create(**kwargs)

    return ChatResult(
        content=list(response.content),
        stop_reason=response.stop_reason,
        usage=response.usage,
        cost_usd=estimate_cost(model, response.usage),
        model=response.model,
    )


# Indirection so tests can substitute a mock.
_chat_fn: Callable[..., ChatResult] = _real_chat


def chat(**kwargs: Any) -> ChatResult:
    return _chat_fn(**kwargs)


def set_chat_fn(fn: Callable[..., ChatResult]) -> None:
    """Tests use this to inject a mock chat function."""
    global _chat_fn
    _chat_fn = fn


def reset_chat_fn() -> None:
    global _chat_fn
    _chat_fn = _real_chat


def require_api_key() -> None:
    """Fail fast with a helpful message if ANTHROPIC_API_KEY is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running an audit:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Get a key at https://console.anthropic.com/"
        )
