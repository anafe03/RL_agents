"""ptscribe — ambient SOAP-note generation for PT/OT/ST visits."""

from ptscribe.eval import run_eval, section_completeness
from ptscribe.hallucination import find_hallucinations
from ptscribe.models import (
    Assessment,
    CheckStatus,
    EvalResult,
    Exercise,
    HallucinationFinding,
    MMTGrade,
    Objective,
    PainScore,
    Plan,
    ROMMeasurement,
    RunRecord,
    SOAPNote,
    Side,
    Subjective,
)
from ptscribe.scribe import extract_soap

__version__ = "0.0.1"

__all__ = [
    "Assessment", "CheckStatus", "EvalResult", "Exercise",
    "HallucinationFinding", "MMTGrade", "Objective", "PainScore",
    "Plan", "ROMMeasurement", "RunRecord", "SOAPNote", "Side", "Subjective",
    "extract_soap", "find_hallucinations", "run_eval", "section_completeness",
]
