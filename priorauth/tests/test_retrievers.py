"""Tests for the pluggable retriever interface + benchmark harness.

Local-only retrievers (BM25, Chroma) are tested for real — no API key
required. LLMJudgedRetriever is tested with the mock chat.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from priorauth import llm
from priorauth.benchmark import load_golden, run_benchmark
from priorauth.mock import make_mock_chat
from priorauth.models import load_case, load_guideline_corpus
from priorauth.retrievers import (
    BM25Retriever,
    LLMJudgedRetriever,
    get_retriever,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "data" / "cases"
GUIDELINES_DIR = ROOT / "data" / "guidelines"
GOLDEN_PATH = ROOT / "data" / "golden.yaml"


@pytest.fixture
def corpus():
    return load_guideline_corpus(GUIDELINES_DIR)


@pytest.fixture
def cases():
    return [load_case(p) for p in sorted(CASES_DIR.glob("*.yaml"))]


@pytest.fixture
def golden():
    return load_golden(GOLDEN_PATH)


def test_bm25_retrieves_relevant_guidelines_for_each_case(corpus, cases, golden):
    r = BM25Retriever()
    r.index(corpus)
    for case in cases:
        expected = set(golden.get(case.id, []))
        got = {g.id for g in r.retrieve(case, k=5)}
        # BM25 should hit at least one expected guideline per case (recall@5 > 0)
        assert got & expected, f"BM25 missed all expected for {case.id}; got {got}"


def test_llm_judged_retriever_with_mock(corpus, cases, golden):
    r = LLMJudgedRetriever()
    r.index(corpus)
    llm.set_chat_fn(make_mock_chat())
    try:
        for case in cases:
            expected = set(golden.get(case.id, []))
            got = {g.id for g in r.retrieve(case, k=5)}
            # Mock returns exactly the expected sets — should be perfect recall
            assert got >= expected, f"LLM-judged mock missed for {case.id}; got {got}, expected ⊇ {expected}"
    finally:
        llm.reset_chat_fn()


def test_registry_constructs_local_retrievers():
    bm25 = get_retriever("bm25")
    assert bm25.name == "bm25"
    assert bm25.category == "keyword"


def test_benchmark_produces_results_for_each_retriever(corpus, cases, golden):
    retrievers = [BM25Retriever()]
    # Add LLM-judged with mock
    llm.set_chat_fn(make_mock_chat())
    try:
        retrievers.append(LLMJudgedRetriever())
        report = run_benchmark(retrievers, cases, corpus, golden, k=5)
    finally:
        llm.reset_chat_fn()

    assert report.k == 5
    assert len(report.retrievers) == 2
    assert "bm25" in report.retrievers
    assert "llm_judged" in report.retrievers
    # Each retriever should have one result per case
    assert len(report.results) == 2 * len(cases)
    # All recall/precision in [0, 1]
    for r in report.results:
        assert 0.0 <= r.precision_at_k <= 1.0
        assert 0.0 <= r.recall_at_k <= 1.0


def test_llm_judged_achieves_perfect_recall_against_mock(corpus, cases, golden):
    """The mock returns exactly the golden IDs, so recall should be 1.0."""
    llm.set_chat_fn(make_mock_chat())
    try:
        report = run_benchmark([LLMJudgedRetriever()], cases, corpus, golden, k=5)
    finally:
        llm.reset_chat_fn()
    by_ret = report.by_retriever()
    assert by_ret["llm_judged"]["recall_at_k"] == 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
