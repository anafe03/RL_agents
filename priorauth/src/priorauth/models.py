"""Pydantic models for PriorAuth Assist.

The whole pipeline is structured: every artifact has a typed model, every
LLM step is constrained to produce a model. Free-text only appears inside
declared string fields, never as the surrounding shape.
"""

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


class PatientContext(BaseModel):
    """De-identified clinical context for one patient under consideration.

    NO PHI — all values are synthetic for the bundled cases. Real use would
    require formal de-identification per HIPAA Safe Harbor or Expert Determination.
    """

    case_id: str
    demographics: str = ""  # e.g. "52-year-old male"
    diagnoses: list[str] = Field(default_factory=list)
    medications_tried: list[str] = Field(default_factory=list)
    relevant_labs: dict[str, str] = Field(default_factory=dict)  # name -> value
    clinical_history: str = ""  # free-text summary, ≤ 1500 chars
    contraindications: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)


class DenialLetter(BaseModel):
    """The insurer's denial as a structured object."""

    payer: str = ""
    member_id: str = ""  # synthetic
    requested_service: str = ""
    denial_reason: str = ""  # e.g. "step therapy not completed"
    cited_policy: str = ""
    raw_text: str = ""  # the original letter body


class Case(BaseModel):
    """One end-to-end case: patient context + denial letter."""

    id: str
    title: str
    requested_service: str
    patient: PatientContext
    denial: DenialLetter


class Guideline(BaseModel):
    """One excerpt from a clinical guideline corpus.

    Each guideline has a stable `id` and an authoritative `source` so the
    drafter can cite it correctly and the assessor can verify the citation.
    """

    id: str  # e.g. "ada_2024_glp1"
    source: str  # e.g. "ADA Standards of Care 2024, Section 9.4"
    organization: str = ""  # e.g. "American Diabetes Association"
    topic: str = ""
    excerpt: str  # the actual text the model is allowed to quote
    tags: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """One claim in the appeal, paired with the guideline that supports it."""

    claim: str  # the clinical assertion in the appeal
    guideline_id: str  # references Guideline.id
    quoted_excerpt: str = ""  # the relevant fragment of the guideline


class Appeal(BaseModel):
    """The drafted appeal letter, structured."""

    id: str = Field(default_factory=_uid)
    case_id: str
    payer: str = ""
    requested_service: str = ""

    opening: str = ""  # 1-2 sentence framing
    clinical_rationale: list[str] = Field(default_factory=list)  # numbered points
    citations: list[Citation] = Field(default_factory=list)
    closing: str = ""  # the formal ask

    drafter_model: str = ""
    drafted_at: datetime = Field(default_factory=_now)


class RubricVerdict(str, Enum):
    EXCELLENT = "excellent"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class AppealAssessment(BaseModel):
    """Independent assessor's verdict on a drafted appeal.

    Different objective from the drafter: the assessor's job is to find weak
    points before the appeal is sent.
    """

    appeal_id: str
    verdict: RubricVerdict
    addressed_all_denial_criteria: bool
    all_claims_cited: bool
    patient_facts_accurate: bool
    has_clear_ask: bool
    reasoning: str = ""  # one-paragraph rationale
    weak_points: list[str] = Field(default_factory=list)  # concrete issues to fix
    cost_usd: float = 0.0


# ---- file loaders ---------------------------------------------------------


def load_case(path: Path | str) -> Case:
    """Load a synthetic case YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    pt = raw.get("patient", {}) or {}
    denial = raw.get("denial", {}) or {}
    return Case(
        id=raw.get("id", Path(path).stem),
        title=raw.get("title", ""),
        requested_service=raw.get("requested_service", ""),
        patient=PatientContext(case_id=raw.get("id", Path(path).stem), **pt),
        denial=DenialLetter(**denial),
    )


def load_guideline_corpus(guidelines_dir: Path | str) -> list[Guideline]:
    """Load every guideline YAML under `guidelines_dir/*.yaml`."""
    guidelines_dir = Path(guidelines_dir)
    if not guidelines_dir.exists():
        return []
    out: list[Guideline] = []
    for p in sorted(guidelines_dir.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text()) or {}
        # Support both single-doc (one guideline per file) and multi-doc
        if isinstance(raw, list):
            for entry in raw:
                out.append(Guideline(**entry))
        else:
            out.append(Guideline(**raw))
    return out
