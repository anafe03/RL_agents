"""Core data models for rehablens."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# MediaPipe's pose model returns 33 landmarks. We only name the ones the
# exercise rules need — the rest stay accessible by index.
class Landmark(int, Enum):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


class PointXY(BaseModel):
    """A normalized image-space landmark point (x, y in 0..1) + visibility."""

    x: float
    y: float
    visibility: float = 1.0

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class PoseFrame(BaseModel):
    """The pose detected in one image — landmarks indexed 0..32."""

    detected: bool
    landmarks: list[PointXY] = Field(default_factory=list)

    def point(self, lm: Landmark) -> PointXY | None:
        idx = int(lm)
        if not self.detected or idx >= len(self.landmarks):
            return None
        return self.landmarks[idx]


class FormStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class FormCheck(BaseModel):
    """One judgment about the patient's form on this frame."""

    name: str  # e.g. "knee depth", "hip drop"
    status: FormStatus
    value: float | None = None  # the measured number (an angle, a delta)
    unit: str = "deg"
    target: str = ""  # human-readable target, e.g. "≥ 90°"
    note: str = ""  # one-line explanation


class SessionResult(BaseModel):
    """The full assessment of one image against one exercise."""

    exercise_id: str
    detected: bool
    checks: list[FormCheck] = Field(default_factory=list)
    summary: str = ""  # one-line overall verdict

    @property
    def overall(self) -> FormStatus:
        if not self.detected or not self.checks:
            return FormStatus.UNKNOWN
        if any(c.status == FormStatus.FAIL for c in self.checks):
            return FormStatus.FAIL
        if any(c.status == FormStatus.WARN for c in self.checks):
            return FormStatus.WARN
        return FormStatus.OK
