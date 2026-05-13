"""Citation verification.

After the extractor produces metrics / tone / surprises / questions, every
quoted excerpt is checked against the transcript text. If a quote is not
a substring of the transcript (case- and whitespace-tolerant), it's
flagged as unverified — meaning the model fabricated the quote.

This is the safety layer that makes the report trustworthy.
"""

from __future__ import annotations

import re

from earningscall.models import (
    AnalystQuestion,
    CitationVerification,
    Metric,
    Quote,
    Surprise,
    ToneAssessment,
    Transcript,
)


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace for tolerant substring matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _quote_in_transcript(quote_text: str, transcript_text: str) -> bool:
    if not quote_text:
        return False
    q = _normalize(quote_text)
    t = _normalize(transcript_text)
    if q in t:
        return True
    # Tolerant: try without leading/trailing quote marks / punctuation
    q_strip = q.strip("\"'.,;:—-—").strip()
    return q_strip != "" and q_strip in t


def verify_quotes(
    transcript: Transcript,
    metrics: list[Metric],
    tone: list[ToneAssessment],
    surprises: list[Surprise],
    questions: list[AnalystQuestion],
) -> list[CitationVerification]:
    """Run every quote through the substring check."""
    text = transcript.full_text
    results: list[CitationVerification] = []

    for m in metrics:
        results.append(_check(m.quote, text, "metrics"))
    for t in tone:
        for q in t.evidence:
            results.append(_check(q, text, f"tone ({t.speaker_name} — {t.segment})"))
    for s in surprises:
        for q in s.evidence:
            results.append(_check(q, text, f"surprises — {s.headline[:40]}"))
    for aq in questions:
        results.append(_check(aq.quote, text, f"analyst Q — {aq.analyst_name}"))

    return results


def _check(quote: Quote, transcript_text: str, where: str) -> CitationVerification:
    return CitationVerification(
        quote_text=quote.text,
        speaker_name=quote.speaker_name,
        found=_quote_in_transcript(quote.text, transcript_text),
        where_used=where,
    )
