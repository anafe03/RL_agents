"""Target output schema for the fine-tuning task.

The model is trained to produce a JSON object that validates against this
Pydantic schema given a free-text patient summary. Designed to match the
shape PriorAuth's drafter would consume.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MedicationOutcome(str, Enum):
    """Why the patient stopped a medication."""

    TRIED_WORKED = "tried_worked"
    TRIED_INTOLERANCE = "tried_intolerance"
    TRIED_INEFFECTIVE = "tried_ineffective"
    TRIED_HYPOGLYCEMIA = "tried_hypoglycemia"
    NOT_TRIED = "not_tried"


class Medication(BaseModel):
    name: str
    outcome: MedicationOutcome
    note: str = ""


class PriorAuthExtraction(BaseModel):
    """The structured output a model must produce from a free-text summary."""

    primary_diagnosis: str
    diagnosis_duration_years: float | None = None
    medications: list[Medication] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    requested_service: str = ""

    model_config = {"extra": "forbid"}  # strict — any extra fields fail validation
