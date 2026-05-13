"""Pydantic models for Earnings Call Inspector."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:8]


class SpeakerRole(str, Enum):
    OPERATOR = "operator"
    CEO = "ceo"
    CFO = "cfo"
    OTHER_EXEC = "other_exec"
    ANALYST = "analyst"
    UNKNOWN = "unknown"


class SpeakerTurn(BaseModel):
    """One speaker turn in the transcript."""

    turn_id: int  # 0-indexed position in transcript
    speaker_name: str
    speaker_role: SpeakerRole
    affiliation: str = ""  # for analysts: "Goldman Sachs", etc.
    text: str
    is_question: bool = False


class Transcript(BaseModel):
    """A loaded earnings transcript."""

    id: str
    company: str
    ticker: str = ""
    period: str = ""  # e.g. "Q4 2025"
    call_date: str = ""  # human-readable date
    sector: str = ""
    turns: list[SpeakerTurn] = Field(default_factory=list)
    raw_text: str = ""  # full original transcript for citation verification

    @property
    def full_text(self) -> str:
        """Joined turn text — used to verify citation substrings."""
        if self.raw_text:
            return self.raw_text
        return "\n\n".join(t.text for t in self.turns)


class Quote(BaseModel):
    """A verbatim quote from the transcript, with speaker attribution."""

    speaker_name: str = ""
    text: str  # the quoted excerpt — must be verifiably a substring of the transcript


class Metric(BaseModel):
    """One financial metric the agent extracted."""

    name: str  # e.g. "Revenue", "Adjusted EPS", "Q4 2025 guidance"
    value: str  # e.g. "$3.42B", "$0.21", "8-10% growth"
    vs_expectations: str = ""  # "beat", "miss", "in-line", "—"
    quote: Quote


class ToneAssessment(BaseModel):
    """Tone/sentiment assessment for one speaker in a defined segment."""

    speaker_name: str
    speaker_role: SpeakerRole
    segment: str = ""  # e.g. "prepared remarks", "Q&A — segment results", "Q&A — guidance"
    sentiment: str  # e.g. "cautiously optimistic", "defensive", "evasive", "confident"
    evidence: list[Quote] = Field(default_factory=list)
    note: str = ""  # one-line analyst-style observation


class Surprise(BaseModel):
    """A notable surprise — guidance revision, new initiative, leadership change, etc."""

    headline: str  # one-line description
    kind: str = ""  # "guidance_revision" | "new_initiative" | "leadership_change" | "macro_callout" | "other"
    significance: str = "medium"  # "low" | "medium" | "high"
    evidence: list[Quote] = Field(default_factory=list)
    rationale: str = ""  # why this matters


class AnalystQuestion(BaseModel):
    """One analyst question, scored by how sharp it was."""

    turn_id: int  # the SpeakerTurn it came from
    analyst_name: str
    affiliation: str = ""
    question_summary: str  # one-line restatement of what they asked
    sharpness: int  # 1..5 — 5 is "actually pushed for a substantive answer"
    answer_quality: str = ""  # "direct" | "hedged" | "dodged" | "promised follow-up"
    quote: Quote
    rationale: str = ""  # why we rated it this way


class CitationVerification(BaseModel):
    """Per-quote citation verification result."""

    quote_text: str
    speaker_name: str = ""
    found: bool  # was the substring actually in the transcript
    where_used: str = ""  # which pass/insight this quote belonged to


class EarningsReport(BaseModel):
    """The full structured output of inspecting a transcript."""

    id: str = Field(default_factory=_uid)
    transcript_id: str
    company: str
    period: str = ""
    metrics: list[Metric] = Field(default_factory=list)
    tone: list[ToneAssessment] = Field(default_factory=list)
    surprises: list[Surprise] = Field(default_factory=list)
    analyst_questions: list[AnalystQuestion] = Field(default_factory=list)
    citation_results: list[CitationVerification] = Field(default_factory=list)
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)

    @property
    def all_citations_verified(self) -> bool:
        return all(c.found for c in self.citation_results)

    @property
    def n_unverified(self) -> int:
        return sum(1 for c in self.citation_results if not c.found)


def load_transcript(path: Path | str) -> Transcript:
    """Load a transcript YAML."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    turns_raw = raw.get("turns", [])
    turns = [SpeakerTurn(**t) for t in turns_raw]
    raw_text = "\n\n".join(t.text for t in turns) if turns else raw.get("raw_text", "")
    return Transcript(
        id=raw.get("id", path.stem),
        company=raw.get("company", ""),
        ticker=raw.get("ticker", ""),
        period=raw.get("period", ""),
        call_date=raw.get("call_date", ""),
        sector=raw.get("sector", ""),
        turns=turns,
        raw_text=raw_text,
    )
