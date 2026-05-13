"""Multi-provider LLM client.

Routes between Anthropic and OpenAI based on the model id prefix. All
projects in this repo use the same chat surface — text in, text out,
with a structured `ChatResult` — so the call sites don't have to know
which provider is being used.

Supported model prefixes:
    claude-*        → Anthropic (Sonnet 4.6, Opus 4.7, Haiku 4.5)
    gpt-*, o1-*, o3-*, o4-*  → OpenAI

Anthropic-specific features (prompt caching, tool use) still work when
a claude-* model is selected. The OpenAI path is intentionally simpler
— text in / text out + token counting. Tool-use call sites should stay
on Anthropic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic


# ---- pricing (USD per token) ----------------------------------------------

_PRICES: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7": {"input": 5e-6, "output": 25e-6},
    "claude-sonnet-4-6": {"input": 3e-6, "output": 15e-6},
    "claude-haiku-4-5": {"input": 1e-6, "output": 5e-6},
    # OpenAI — for cheap testing / cross-provider validation
    "gpt-4o": {"input": 2.5e-6, "output": 1e-5},
    "gpt-4o-mini": {"input": 1.5e-7, "output": 6e-7},   # ~20x cheaper than Sonnet
    "gpt-4.1": {"input": 2e-6, "output": 8e-6},
    "gpt-4.1-mini": {"input": 4e-7, "output": 1.6e-6},
}


SUPPORTED_MODELS: list[str] = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
    "gpt-4o",
    "gpt-4o-mini",
]


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


# ---- result type ----------------------------------------------------------

@dataclass
class ChatResult:
    """Provider-agnostic chat response.

    `usage` is whatever the underlying SDK returned (typed differently per
    provider). Use `cost_usd` for the normalized cost. `text` is the
    extracted assistant message.
    """
    text: str
    usage: Any
    cost_usd: float
    model: str


# ---- cost estimation (handles both providers' usage shapes) ---------------

def estimate_cost(model: str, usage: Any) -> float:
    """Estimate USD cost from a usage object.

    Accepts Anthropic's `anthropic.types.Usage` (input_tokens / output_tokens /
    cache_creation_input_tokens / cache_read_input_tokens) and OpenAI's
    `openai.types.CompletionUsage` (prompt_tokens / completion_tokens).
    """
    price = _PRICES.get(model, _PRICES["claude-sonnet-4-6"])
    inp = price["input"]
    out = price["output"]
    # Anthropic shape
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


# ---- Anthropic path -------------------------------------------------------

def _get_anthropic_client() -> anthropic.Anthropic:
    # Do NOT cache — Streamlit sets ANTHROPIC_API_KEY at button-click time,
    # and a cached client created before the key was set silently retains
    # api_key=None and 500s with "Could not resolve authentication method".
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _anthropic_chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 2048,
) -> ChatResult:
    client = _get_anthropic_client()
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


# ---- OpenAI path ----------------------------------------------------------

def _get_openai_client():
    # Same caching gotcha as Anthropic — rebuild each call.
    import openai
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _normalize_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic's `messages` accepts content as either a string or a list of
    content blocks. OpenAI accepts either too, but block format differs. This
    flattens any list-content into plain text — fine for text-only call sites.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        role = m.get("role", "user")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            out.append({"role": role, "content": "\n".join(text_parts)})
        else:
            out.append({"role": role, "content": str(content) if content else ""})
    return out


def _openai_chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 2048,
) -> ChatResult:
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
    return ChatResult(
        text=text,
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
    global _chat_fn
    _chat_fn = fn


def reset_chat_fn() -> None:
    global _chat_fn
    _chat_fn = _real_chat


def require_api_key(model: str | None = None) -> None:
    """Raise if the relevant provider's API key is missing.

    If a model is provided, check the key for THAT provider. If not
    provided, require Anthropic (the default).
    """
    if model and _is_openai_model(model):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                f"OPENAI_API_KEY is not set, but model={model} requires it. "
                "Export OPENAI_API_KEY=sk-... or pick a claude-* model."
            )
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before running real drafts, "
            "or pick a gpt-* model (requires OPENAI_API_KEY instead)."
        )
