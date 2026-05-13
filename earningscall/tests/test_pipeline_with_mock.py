"""End-to-end mock pipeline test for Earnings Call Inspector.

Runs all 4 extraction passes + citation verification on each bundled
transcript using the mock chat. Verifies:
- pipeline returns non-empty results across passes
- every cited quote actually appears in the transcript text
"""

from __future__ import annotations

from pathlib import Path

from earningscall import llm
from earningscall.extractor import (
    extract_analyst_questions,
    extract_metrics,
    extract_surprises,
    extract_tone,
)
from earningscall.mock import make_mock_chat
from earningscall.models import load_transcript
from earningscall.verifier import verify_quotes

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = sorted((ROOT / "data" / "transcripts").glob("*.yaml"))


def _run(path: Path):
    transcript = load_transcript(path)
    llm.set_chat_fn(make_mock_chat())
    try:
        metrics, _ = extract_metrics(transcript)
        tone, _ = extract_tone(transcript)
        surprises, _ = extract_surprises(transcript)
        questions, _ = extract_analyst_questions(transcript)
        citations = verify_quotes(transcript, metrics, tone, surprises, questions)
    finally:
        llm.reset_chat_fn()
    return transcript, metrics, tone, surprises, questions, citations


def test_all_transcripts_produce_results():
    assert TRANSCRIPTS, "expected bundled transcripts"
    for p in TRANSCRIPTS:
        t, metrics, tone, surprises, questions, citations = _run(p)
        assert metrics, f"no metrics for {t.id}"
        assert tone, f"no tone for {t.id}"
        assert surprises, f"no surprises for {t.id}"
        assert questions, f"no questions for {t.id}"
        assert citations, f"no citations for {t.id}"


def test_all_quotes_are_substrings_of_transcript():
    """Citation enforcement — the headline safety property."""
    for p in TRANSCRIPTS:
        t, metrics, tone, surprises, questions, citations = _run(p)
        unverified = [c for c in citations if not c.found]
        assert not unverified, (
            f"{t.id}: {len(unverified)} quotes failed substring check: "
            + "; ".join(f"\"{c.quote_text[:80]}\" in {c.where_used}" for c in unverified[:3])
        )


def test_analyst_question_sharpness_is_in_range():
    for p in TRANSCRIPTS:
        _, _, _, _, questions, _ = _run(p)
        for q in questions:
            assert 1 <= q.sharpness <= 5, f"sharpness out of range: {q.sharpness}"


if __name__ == "__main__":
    test_all_transcripts_produce_results()
    test_all_quotes_are_substrings_of_transcript()
    test_analyst_question_sharpness_is_in_range()
    print("ok")
