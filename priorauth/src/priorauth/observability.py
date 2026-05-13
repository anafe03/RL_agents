"""Optional LangSmith observability for PriorAuth.

LangSmith is great for LLM-app observability — every call traced, inputs +
outputs + latency + cost captured, eval runs over datasets, prompt
versioning. It is NOT inherently a HIPAA-compliance solution; deploying
LangSmith in a HIPAA-compliant way requires a BAA + self-hosted setup +
appropriate access controls. But for general LLM-ops observability it's
the standard tool.

Integration shape: a `@traced(name)` decorator that delegates to
`langsmith.traceable` when LangSmith is installed + configured, and is a
no-op otherwise. Wrap any LLM call site to get tracing for free.

## Setup

```bash
pip install langsmith
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=priorauth-dev
```

Then PriorAuth's existing call sites — the retrievers, drafter, assessor —
auto-trace into your LangSmith project. No code changes needed beyond
applying `@traced(...)` to the entry points (already done in this module
as examples — wire them in where useful).

## Combining with de-identification

If you're running with `Deidentifier()` (see `priorauth.deidentify`), the
LLM call payloads in LangSmith will already contain the tokenized form —
no PHI in the LangSmith logs either. That's the intended composition.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


_TRACING_AVAILABLE: bool | None = None


def _has_langsmith() -> bool:
    """Cached check for whether langsmith is installed AND tracing is enabled."""
    global _TRACING_AVAILABLE
    if _TRACING_AVAILABLE is not None:
        return _TRACING_AVAILABLE
    if not os.environ.get("LANGCHAIN_API_KEY"):
        _TRACING_AVAILABLE = False
        return False
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() not in {"true", "1"}:
        _TRACING_AVAILABLE = False
        return False
    try:
        import langsmith  # noqa: F401
        _TRACING_AVAILABLE = True
    except ImportError:
        _TRACING_AVAILABLE = False
    return _TRACING_AVAILABLE


def traced(name: str | None = None, tags: list[str] | None = None) -> Callable[[F], F]:
    """Decorator: wrap a function with langsmith.traceable if available.

    Falls back to identity if langsmith isn't installed or `LANGCHAIN_API_KEY`
    isn't set — so this is safe to apply unconditionally to LLM call sites.
    """
    def _outer(fn: F) -> F:
        if not _has_langsmith():
            return fn
        from langsmith import traceable
        wrapped = traceable(name=name or fn.__name__, tags=tags)(fn)

        @wraps(fn)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            return wrapped(*args, **kwargs)

        return _inner  # type: ignore[return-value]
    return _outer


# Example wiring — uncomment in your call sites to trace specific entry points:
#
#   from priorauth.observability import traced
#
#   @traced(name="priorauth.drafter.draft_appeal", tags=["drafter"])
#   def draft_appeal(case, guidelines, model="..."):
#       ...
#
# After running, visit https://smith.langchain.com/ to inspect the traces.
