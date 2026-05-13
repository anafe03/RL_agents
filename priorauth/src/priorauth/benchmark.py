"""Retriever benchmark harness.

For each retriever × each case, measure:
- precision@k  : (retrieved ∩ expected) / k
- recall@k     : (retrieved ∩ expected) / |expected|
- latency_ms   : wall-clock time of the retrieve() call
- cost_usd     : accumulated API cost (0 for local retrievers)

Output is a `BenchmarkReport` you can render in the terminal or Streamlit.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from priorauth.models import Case, Guideline
from priorauth.retrievers.base import Retriever


def load_golden(path: Path | str) -> dict[str, list[str]]:
    """Load the per-case expected-guideline-IDs map."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {k: list(v) for k, v in raw.items()}


class CaseResult(BaseModel):
    case_id: str
    retriever_name: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    latency_ms: float
    cost_usd: float
    k: int


class BenchmarkReport(BaseModel):
    k: int
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def retrievers(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.retriever_name not in seen:
                seen.append(r.retriever_name)
        return seen

    @property
    def case_ids(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.case_id not in seen:
                seen.append(r.case_id)
        return seen

    def by_retriever(self) -> dict[str, dict[str, float]]:
        """Aggregate metrics per retriever (mean across cases)."""
        agg: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for r in self.results:
            a = agg.setdefault(
                r.retriever_name,
                {"precision_at_k": 0.0, "recall_at_k": 0.0, "latency_ms": 0.0, "cost_usd": 0.0},
            )
            a["precision_at_k"] += r.precision_at_k
            a["recall_at_k"] += r.recall_at_k
            a["latency_ms"] += r.latency_ms
            a["cost_usd"] += r.cost_usd
            counts[r.retriever_name] = counts.get(r.retriever_name, 0) + 1
        for name, totals in agg.items():
            n = max(1, counts[name])
            totals["precision_at_k"] /= n
            totals["recall_at_k"] /= n
            totals["latency_ms"] /= n  # mean per case
            # cost stays as sum across cases — total run cost is what users care about
        return agg


def _score_one(
    retriever: Retriever,
    case: Case,
    expected_ids: list[str],
    k: int,
) -> CaseResult:
    retriever.reset_cost()
    start = time.perf_counter()
    retrieved = retriever.retrieve(case, k=k)
    latency_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = [g.id for g in retrieved]
    expected_set = set(expected_ids)
    hits = sum(1 for rid in retrieved_ids if rid in expected_set)
    precision = hits / max(1, len(retrieved_ids))
    recall = hits / max(1, len(expected_set))
    return CaseResult(
        case_id=case.id,
        retriever_name=retriever.name,
        retrieved_ids=retrieved_ids,
        expected_ids=expected_ids,
        precision_at_k=precision,
        recall_at_k=recall,
        latency_ms=latency_ms,
        cost_usd=retriever.cost_usd,
        k=k,
    )


def run_benchmark(
    retrievers: list[Retriever],
    cases: list[Case],
    guidelines: list[Guideline],
    golden: dict[str, list[str]],
    k: int = 5,
) -> BenchmarkReport:
    """Run every retriever on every case and produce a BenchmarkReport."""
    report = BenchmarkReport(k=k)
    for retriever in retrievers:
        retriever.index(guidelines)
        for case in cases:
            expected = golden.get(case.id, [])
            if not expected:
                continue
            report.results.append(_score_one(retriever, case, expected, k))
    return report
