"""Core data models for AutoFill."""

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


class ComplaintInput(BaseModel):
    """The structured complaint the agent will file.

    Designed to align with what a PriorAuth `Appeal` already produces, so the
    two projects can be chained.
    """

    id: str
    complainant_name: str = ""
    complainant_email: str = ""
    complainant_phone: str = ""
    complainant_address: str = ""
    insurer_name: str
    insurer_member_id: str = ""
    claim_number: str = ""
    denial_reason: str
    requested_service: str = ""
    narrative: str  # the body of the complaint — pasted into the form's free-text field
    supporting_docs: list[str] = Field(default_factory=list)  # filenames the user could attach


class FormTarget(BaseModel):
    """Description of a public complaint form the agent knows how to fill."""

    id: str
    name: str  # display name, e.g. "California Department of Insurance — Consumer Complaint"
    state: str = ""
    url: str  # the public form URL
    notes: str = ""  # operator notes — known quirks, hidden fields, etc.
    # A small map from semantic field categories to one or more hints
    # the agent's system prompt can use ("look for a field labeled X").
    field_hints: dict[str, list[str]] = Field(default_factory=dict)


class StepAction(str, Enum):
    NAVIGATE = "navigate"        # go to URL
    SCREENSHOT = "screenshot"    # take a screenshot (always done implicitly)
    CLICK = "click"
    TYPE = "type"
    KEY = "key"                  # keyboard shortcut
    SCROLL = "scroll"
    OBSERVE = "observe"          # narration / reasoning step, no real action
    HALT = "halt"                # explicit stop before submit
    SUBMIT_ATTEMPTED = "submit_attempted"  # the agent attempted submit (only in --no-dry-run)


class SubmissionStep(BaseModel):
    """One step in a submission run — for recording, playback, and audit."""

    step_id: int  # 0-indexed
    action: StepAction
    narration: str = ""  # human-readable summary of what the agent is doing / why
    target_label: str = ""  # e.g. "First Name field", "Submit button"
    value: str = ""  # the text typed / coordinate / key — depends on action
    coordinate: tuple[int, int] | None = None
    screenshot_path: str = ""  # filename of the screenshot captured AFTER this step
    timestamp: datetime = Field(default_factory=_now)


class SubmissionResult(BaseModel):
    """The full record of a submission run."""

    id: str = Field(default_factory=_uid)
    complaint_id: str
    target_id: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    steps: list[SubmissionStep] = Field(default_factory=list)
    dry_run: bool = True
    completed_to_review: bool = False  # reached the "ready to submit" state successfully
    submitted: bool = False  # only true in --no-dry-run mode
    error: str = ""
    cost_usd: float = 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)


def load_complaint(path: Path | str) -> ComplaintInput:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    raw.setdefault("id", path.stem)
    return ComplaintInput(**raw)
