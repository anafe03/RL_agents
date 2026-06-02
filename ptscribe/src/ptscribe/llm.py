"""Multi-provider LLM client — Anthropic + OpenAI, uncached.

Same shape as the rest of the repo: text in, text out, with a structured
`ChatResult`. The clients are intentionally NOT cached at module load
because the Streamlit UI sets the API key at button-click time and a
cached client built before the env var was set would silently retain
`api_key=None` and fail at request time.
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
    "gpt-4o": {"input": 2.5e-6, "output": 1e-5},
    "gpt-4o-mini": {"input": 1.5e-7, "output": 6e-7},
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


@dataclass
class ChatResult:
    text: str
    usage: Any
    cost_usd: float
    model: str
    latency_ms: int = 0


def estimate_cost(model: str, usage: Any) -> float:
    price = _PRICES.get(model, _PRICES["claude-sonnet-4-6"])
    input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = (
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    )
    return input_tokens * price["input"] + output_tokens * price["output"]


def _get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _get_openai_client():
    import openai

    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _anthropic_chat(
    *, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int = 2048
) -> ChatResult:
    import time

    client = _get_anthropic_client()
    t0 = time.perf_counter()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=system, messages=messages
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return ChatResult(
        text=text.strip(),
        usage=response.usage,
        cost_usd=estimate_cost(model, response.usage),
        model=response.model,
        latency_ms=latency_ms,
    )


def _openai_chat(
    *, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int = 2048
) -> ChatResult:
    import time

    client = _get_openai_client()
    t0 = time.perf_counter()
    oai_messages = [{"role": "system", "content": system}, *messages]
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=oai_messages
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = (response.choices[0].message.content or "").strip()
    return ChatResult(
        text=text,
        usage=response.usage,
        cost_usd=estimate_cost(model, response.usage),
        model=response.model,
        latency_ms=latency_ms,
    )


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


def require_api_key(model: str) -> None:
    if _is_openai_model(model):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY is not set, but model={model} needs it.")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(f"ANTHROPIC_API_KEY is not set, but model={model} needs it.")
