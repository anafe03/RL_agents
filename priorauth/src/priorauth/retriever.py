"""Backward-compatibility shim.

v0.1 had a single `retrieve_relevant(case, corpus) -> (guidelines, cost)`
function in this module. v0.2 generalizes retrieval behind the
`Retriever` interface in `priorauth.retrievers`. This module preserves
the old function signature so existing callers (the old CLI path, the
Streamlit UI, the v0.1 pipeline test) keep working.

New code should construct a `Retriever` directly:

    from priorauth.retrievers import get_retriever
    r = get_retriever("llm_judged")
    r.index(corpus)
    results = r.retrieve(case, k=5)
"""

from __future__ import annotations

from priorauth.models import Case, Guideline
from priorauth.retrievers.llm_judged import LLMJudgedRetriever


def retrieve_relevant(
    case: Case,
    corpus: list[Guideline],
    model: str = "claude-sonnet-4-6",
) -> tuple[list[Guideline], float]:
    """Backward-compatible wrapper. New code should use `Retriever` directly."""
    r = LLMJudgedRetriever(model=model)
    r.index(corpus)
    results = r.retrieve(case, k=5)
    return results, r.cost_usd
