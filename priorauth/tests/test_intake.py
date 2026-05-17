"""Tests for the pasted-denial intake — extracting a Case from a denial letter.

The intake LLM step is exercised with the mock chat. The extracted case is
then run through the full retriever → drafter → assessor pipeline to confirm
a pasted denial produces a real, citation-checked appeal.
"""

from __future__ import annotations

from pathlib import Path

from priorauth import llm
from priorauth.assessor import assess_appeal
from priorauth.drafter import draft_appeal
from priorauth.intake import coerce_case, extract_case
from priorauth.mock import make_mock_chat
from priorauth.models import load_guideline_corpus
from priorauth.retriever import retrieve_relevant

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_denials" / "orthotic_coding.txt"
GUIDELINES_DIR = ROOT / "data" / "guidelines"


def test_sample_denial_letter_exists():
    assert SAMPLE.exists(), "bundled orthotic sample denial is missing"
    text = SAMPLE.read_text()
    assert "Meridian Health Plan" in text
    assert "L3000" in text


def test_extract_case_from_orthotic_letter():
    llm.set_chat_fn(make_mock_chat())
    try:
        case, cost = extract_case(SAMPLE.read_text(), "")
    finally:
        llm.reset_chat_fn()
    assert case.denial.payer == "Meridian Health Plan"
    assert case.denial.member_id == "MHP-4471902-C"
    assert "L3000" in case.requested_service
    assert case.patient.diagnoses, "expected the intake to extract diagnoses"
    assert case.denial.raw_text.strip(), "raw denial letter should be preserved"
    assert cost == 0.0  # mock is free


def test_coerce_case_handles_minimal_json():
    case = coerce_case(
        '{"title": "T", "requested_service": "X", "patient": {}, "denial": {}}',
        "the raw letter",
    )
    assert case.title == "T"
    assert case.denial.raw_text == "the raw letter"


def test_coerce_case_extracts_json_from_prose():
    case = coerce_case(
        'Sure: {"title": "Y", "patient": {}, "denial": {"payer": "Acme"}} — done',
        "L",
    )
    assert case.denial.payer == "Acme"


def test_coerce_case_drops_unknown_fields():
    # Unexpected keys in the LLM reply must not crash model construction.
    case = coerce_case(
        '{"title": "T", "patient": {"demographics": "adult", "bogus": 1}, '
        '"denial": {"payer": "P", "junk": "x"}}',
        "L",
    )
    assert case.patient.demographics == "adult"
    assert case.denial.payer == "P"


def test_extracted_case_runs_full_pipeline():
    llm.set_chat_fn(make_mock_chat())
    try:
        case, _ = extract_case(SAMPLE.read_text(), "")
        corpus = load_guideline_corpus(GUIDELINES_DIR)
        selected, _ = retrieve_relevant(case, corpus)
        appeal, _ = draft_appeal(case, selected)
        assessment = assess_appeal(case, appeal, selected)
    finally:
        llm.reset_chat_fn()
    assert selected, "retriever found no guidelines for the pasted orthotic case"
    assert appeal.citations, "the pasted-case appeal has no citations"
    # Citation enforcement: every cited guideline must be in the retrieved set.
    valid_ids = {g.id for g in selected}
    for cit in appeal.citations:
        assert cit.guideline_id in valid_ids, f"appeal cites un-retrieved {cit.guideline_id}"
    assert assessment.verdict is not None


if __name__ == "__main__":
    test_sample_denial_letter_exists()
    test_extract_case_from_orthotic_letter()
    test_coerce_case_handles_minimal_json()
    test_coerce_case_extracts_json_from_prose()
    test_coerce_case_drops_unknown_fields()
    test_extracted_case_runs_full_pipeline()
    print("ok")
