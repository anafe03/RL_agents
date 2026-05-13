"""Retriever interface — every retrieval backend implements this.

The interface is deliberately narrow:
- `name` so the benchmark can label results
- `category` so we can group ("keyword", "dense_vector", "llm_judged")
- `index(guidelines)` — one-time setup
- `retrieve(case, k)` — return up to k Guideline objects relevant to the case
- `cost_usd` — running total of any API costs (LLM-judged) or 0 for local
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from priorauth.models import Case, Guideline

RetrieverCategory = Literal["keyword", "dense_vector", "llm_judged"]


class Retriever(ABC):
    """Abstract base — implementations: BM25Retriever, ChromaRetriever, LLMJudgedRetriever."""

    name: str
    category: RetrieverCategory

    def __init__(self) -> None:
        self.cost_usd: float = 0.0

    @abstractmethod
    def index(self, guidelines: list[Guideline]) -> None:
        """Build the retrieval index from a guideline corpus. Idempotent."""

    @abstractmethod
    def retrieve(self, case: Case, k: int = 5) -> list[Guideline]:
        """Return up to k guidelines most relevant to this case, best-first."""

    def reset_cost(self) -> None:
        self.cost_usd = 0.0


def case_to_query(case: Case) -> str:
    """Convert a structured case into a single query string for retrieval.

    Concatenates the denial reason, requested service, diagnoses, contraindications,
    and red flags — the fields most likely to drive relevant guideline matching.
    """
    p = case.patient
    parts: list[str] = [
        f"Requested service: {case.requested_service}",
        f"Denial reason: {case.denial.denial_reason}",
        "Diagnoses: " + "; ".join(p.diagnoses),
        "Medications tried: " + "; ".join(p.medications_tried),
    ]
    if p.contraindications:
        parts.append("Contraindications: " + "; ".join(p.contraindications))
    if p.red_flags:
        parts.append("Red flags: " + "; ".join(p.red_flags))
    return " | ".join(parts)
