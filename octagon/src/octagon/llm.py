"""Multi-provider LLM client for Octagon.

Octagon's defender uses Anthropic-format tool use (the audit pipeline depends
on `tool_use` content blocks). The judge does NOT use tools — pure JSON
output. So the multi-provider split is:

  - Anthropic models (claude-*): full support, tools + prompt caching.
  - OpenAI models (gpt-*, o*-*): supported ONLY when `tools` is empty/None.
    Useful for running the judge on a cheap GPT-4o-mini for fast iteration.
    Attempting to use an OpenAI model with `tools` raises a clear error.

The result of this asymmetry: defender stays on Anthropic; judge can be
swapped to GPT-4o-mini and save ~95% on judge cost during iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic


_PRICES: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7": {"input": 5e-6, "output": 25e-6},
    "claude-opus-4-6": {"input": 5e-6, "output": 25e-6},
    "claude-sonnet-4-6": {"input": 3e-6, "output": 15e-6},
    "claude-haiku-4-5": {"input": 1e-6, "output": 5e-6},
    # OpenAI
    "gpt-4o": {"input": 2.5e-6, "output": 1e-5},
    "gpt-4o-mini": {"input": 1.5e-7, "output": 6e-7},
    "gpt-4.1": {"input": 2e-6, "output": 8e-6},
    "gpt-4.1-mini": {"input": 4e-7, "output": 1.6e-6},
}


# What the UI / CLI exposes for selection. Defender call sites should ONLY
# select from the Anthropic models (Octagon's defender uses tools); judge
# call sites can use anything in this list.
SUPPORTED_MODELS: list[str] = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
    "gpt-4o",
    "gpt-4o-mini",
]


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def estimate_cost(model: str, usage: Any) -> float:
    """USD cost from a usage object (Anthropic or OpenAI shape)."""
    price = _PRICES.get(model, _PRICES["claude-sonnet-4-6"])
    inp = price["input"]
    out = price["output"]
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        input_tokens * inp
        + output_tokens * out
        + cache_write * (inp * 1.25)
        + cache_read * (inp * 0.1)
    )


@dataclass
class ChatResult:
    """Normalized response from a single chat call."""

    content: list[Any]  # list of content blocks (text, tool_use, etc.)
    stop_reason: str | None
    usage: Any
    cost_usd: float
    model: str


# ---- Anthropic path -------------------------------------------------------

_anth_client: anthropic.Anthropic | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anth_client
    if _anth_client is None:
        _anth_client = anthropic.Anthropic()
    return _anth_client


def _anthropic_chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    thinking_mode: str = "disabled",
    effort: str | None = None,
) -> ChatResult:
    client = _get_anthropic_client()
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


# ---- OpenAI path (no tools — judge-only use) ------------------------------

_oai_client = None


def _get_openai_client():
    global _oai_client
    if _oai_client is None:
        import openai
        _oai_client = openai.OpenAI()
    return _oai_client


def _normalize_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        role = m.get("role", "user")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            out.append({"role": role, "content": "\n".join(text_parts)})
        else:
            out.append({"role": role, "content": str(content) if content else ""})
    return out


def _openai_chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    thinking_mode: str = "disabled",
    effort: str | None = None,
) -> ChatResult:
    if tools:
        raise NotImplementedError(
            f"OpenAI tool-use is not implemented in Octagon's LLM client. "
            f"The defender path requires Anthropic-format tool blocks. "
            f"Select a claude-* model for the defender; OpenAI models work "
            f"for the judge (which doesn't use tools)."
        )
    client = _get_openai_client()
    oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    oai_messages.extend(_normalize_messages_for_openai(messages))
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=oai_messages,
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    # Synthesize an Anthropic-shaped content list so call sites that iterate
    # `result.content` looking for `type == "text"` blocks keep working.
    synthesized_blocks = [type("Block", (), {"type": "text", "text": text})()]
    return ChatResult(
        content=synthesized_blocks,
        stop_reason="end_turn",
        usage=response.usage,
        cost_usd=estimate_cost(model, response.usage),
        model=response.model,
    )


# ---- router --------------------------------------------------------------

def _real_chat(*, model: str, **kwargs: Any) -> ChatResult:
    if _is_openai_model(model):
        return _openai_chat(model=model, **kwargs)
    return _anthropic_chat(model=model, **kwargs)


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


def require_api_key(model: str | None = None) -> None:
    if model and _is_openai_model(model):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                f"OPENAI_API_KEY is not set, but model={model} requires it."
            )
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running an audit:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Get a key at https://console.anthropic.com/"
        )
