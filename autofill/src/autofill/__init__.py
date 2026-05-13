"""AutoFill — Computer Use agent that fills public insurance complaint forms."""

from autofill.models import (
    ComplaintInput,
    FormTarget,
    StepAction,
    SubmissionResult,
    SubmissionStep,
    load_complaint,
)
from autofill.targets import REGISTRY, get_target

__version__ = "0.0.1"

__all__ = [
    "ComplaintInput",
    "FormTarget",
    "REGISTRY",
    "StepAction",
    "SubmissionResult",
    "SubmissionStep",
    "get_target",
    "load_complaint",
]
