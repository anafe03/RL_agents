"""End-to-end mock pipeline test for PriorAuth Assist.

Runs retriever → drafter → assessor on each bundled case using the mock chat.
Verifies citation enforcement: every citation references a guideline ID that
was in the retriever's selected set.
"""

from __future__ import annotations

from pathlib import Path

from priorauth import llm
from priorauth.assessor import assess_appeal
from priorauth.drafter import draft_appeal
from priorauth.mock import make_mock_chat
from priorauth.models import load_case, load_guideline_corpus
from priorauth.retriever import retrieve_relevant

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "data" / "cases").glob("*.yaml"))
GUIDELINES_DIR = ROOT / "data" / "guidelines"


def _run_case(case_path: Path):
    case = load_case(case_path)
    corpus = load_guideline_corpus(GUIDELINES_DIR)
    llm.set_chat_fn(make_mock_chat())
    try:
        selected, _ = retrieve_relevant(case, corpus)
        appeal, _ = draft_appeal(case, selected)
        assessment = assess_appeal(case, appeal, selected)
    finally:
        llm.reset_chat_fn()
    return case, selected, appeal, assessment


def test_pipeline_runs_for_all_cases():
    assert CASES, "expected bundled cases under data/cases/"
    for case_path in CASES:
        case, selected, appeal, assessment = _run_case(case_path)
        assert selected, f"retriever returned no guidelines for {case.id}"
        assert appeal.clinical_rationale, f"appeal has no rationale for {case.id}"
        assert appeal.citations, f"appeal has no citations for {case.id}"
        assert appeal.opening, f"appeal missing opening for {case.id}"
        assert appeal.closing, f"appeal missing closing for {case.id}"


def test_citations_only_reference_retrieved_guidelines():
    """Citation enforcement: every cited guideline_id must be in the retrieved set."""
    for case_path in CASES:
        case, selected, appeal, _ = _run_case(case_path)
        valid_ids = {g.id for g in selected}
        for cit in appeal.citations:
            assert cit.guideline_id in valid_ids, (
                f"{case.id}: appeal cites '{cit.guideline_id}' which was not in the retrieved set {valid_ids}"
            )


def test_assessor_produces_verdict():
    for case_path in CASES:
        case, _, _, assessment = _run_case(case_path)
        assert assessment.verdict is not None, f"missing verdict for {case.id}"
        assert assessment.reasoning, f"missing assessor reasoning for {case.id}"


if __name__ == "__main__":
    test_pipeline_runs_for_all_cases()
    test_citations_only_reference_retrieved_guidelines()
    test_assessor_produces_verdict()
    print("ok")
