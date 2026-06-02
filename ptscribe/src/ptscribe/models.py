"""Pydantic models for ptscribe.

The whole point of structured extraction: every PT-specific datum
(ROM degrees, MMT grades, pain score, named exercise) lives in a typed
field so the eval harness can audit each one against the source
transcript independently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# -- PT-specific measurement types -------------------------------------------


class JointMovement(str, Enum):
    """Common PT range-of-motion movements."""

    KNEE_FLEXION = "knee flexion"
    KNEE_EXTENSION = "knee extension"
    HIP_FLEXION = "hip flexion"
    HIP_ABDUCTION = "hip abduction"
    SHOULDER_FLEXION = "shoulder flexion"
    SHOULDER_ABDUCTION = "shoulder abduction"
    ANKLE_DORSIFLEXION = "ankle dorsiflexion"
    LUMBAR_FLEXION = "lumbar flexion"
    CERVICAL_ROTATION = "cervical rotation"
    OTHER = "other"


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    UNSPECIFIED = "unspecified"


class ROMMeasurement(BaseModel):
    """One range-of-motion observation, e.g. 'right knee flexion 110°'."""

    joint: str  # free-form when JointMovement.OTHER, otherwise the enum value
    side: Side = Side.UNSPECIFIED
    degrees: float
    description: str = ""  # e.g. "with mild pain at end-range"


class MMTGrade(BaseModel):
    """Manual muscle test on the 0..5 scale."""

    muscle_group: str  # e.g. "quadriceps", "deltoid"
    side: Side = Side.UNSPECIFIED
    grade: float  # 0..5, allow 0.5 increments
    note: str = ""


class PainScore(BaseModel):
    """Self-reported pain, 0..10."""

    location: str  # e.g. "right knee", "low back"
    score: int  # 0..10
    when: str = ""  # e.g. "at rest", "with stairs", "after activity"


class Exercise(BaseModel):
    """A prescribed therapeutic exercise."""

    name: str  # e.g. "straight-leg raise", "wall slide"
    sets: int | None = None
    reps: int | None = None
    hold_seconds: int | None = None
    sessions_per_day: int | None = None
    note: str = ""  # e.g. "to tolerance", "as pain allows"


# -- SOAP-note structure -----------------------------------------------------


class Subjective(BaseModel):
    """The S of SOAP — patient-reported information."""

    chief_complaint: str = ""
    history: str = ""  # 1-3 sentence narrative
    pain: list[PainScore] = Field(default_factory=list)
    functional_limitations: list[str] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)


class Objective(BaseModel):
    """The O of SOAP — measurable clinician observations."""

    rom: list[ROMMeasurement] = Field(default_factory=list)
    strength: list[MMTGrade] = Field(default_factory=list)
    special_tests: list[str] = Field(default_factory=list)  # free-form
    observations: list[str] = Field(default_factory=list)   # gait, posture, etc.


class Assessment(BaseModel):
    """The A of SOAP — clinician's interpretation."""

    summary: str = ""  # 2-4 sentence narrative
    progress: str = ""  # improving / stable / regressing — free-form
    impairments: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """The P of SOAP — next steps."""

    interventions_today: list[str] = Field(default_factory=list)
    home_exercise_program: list[Exercise] = Field(default_factory=list)
    next_visit: str = ""  # frequency / when
    referrals: list[str] = Field(default_factory=list)


class SOAPNote(BaseModel):
    """A full structured SOAP note for a PT/OT/ST visit."""

    patient_label: str = ""  # de-identified label, e.g. "Patient A — post-op TKA"
    visit_type: str = ""  # e.g. "initial eval", "progress note", "post-op visit"
    discipline: str = "PT"  # PT, OT, ST
    subjective: Subjective = Field(default_factory=Subjective)
    objective: Objective = Field(default_factory=Objective)
    assessment: Assessment = Field(default_factory=Assessment)
    plan: Plan = Field(default_factory=Plan)


# -- eval / monitoring -------------------------------------------------------


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class HallucinationFinding(BaseModel):
    """One claim in the SOAP note that we could not ground in the transcript."""

    field_path: str  # e.g. "objective.rom[0]"
    claim: str       # the human-readable claim
    confidence: float = 0.0  # 0..1 — how confident the check is that this is a fab
    note: str = ""


class EvalResult(BaseModel):
    """Eval-harness output for one scribe run."""

    transcript_id: str = ""
    has_all_sections: bool = False
    completeness_score: float = 0.0  # 0..1 — how much of the expected structure is filled
    hallucination_findings: list[HallucinationFinding] = Field(default_factory=list)
    judge_score: float | None = None  # LLM-as-judge for narrative quality, 0..1
    judge_reasoning: str = ""
    overall: CheckStatus = CheckStatus.WARN

    @property
    def hallucination_rate(self) -> float:
        return len(self.hallucination_findings) / 10  # cap for display purposes


class RunRecord(BaseModel):
    """One end-to-end scribe run — what monitoring.py logs."""

    id: str | None = None  # filled by the DB
    timestamp: datetime = Field(default_factory=_now)
    transcript_id: str = ""
    model: str = ""
    mode: str = "demo"  # "demo" or "live"
    cost_usd: float = 0.0
    latency_ms: int = 0
    input_chars: int = 0
    output_chars: int = 0
    completeness_score: float = 0.0
    hallucination_count: int = 0
    judge_score: float | None = None
    error: str = ""


# -- loaders -----------------------------------------------------------------


def load_transcript(path: Path | str) -> tuple[str, str]:
    """Read a transcript file. Returns (transcript_id, text)."""
    path = Path(path)
    return path.stem, path.read_text()


def load_golden(path: Path | str) -> dict[str, Any]:
    """Load a golden expected-SOAP-note YAML for the eval harness."""
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}
