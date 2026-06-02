"""The eval harness — section completeness + hallucination + LLM-as-judge.

This is the centerpiece. Every run produces a `EvalResult` summarizing:
  - whether the four SOAP sections are populated
  - a 0..1 completeness score across expected fields
  - the list of claims that didn't ground in the transcript
  - an optional LLM-as-judge score for the narrative sections

The judge is opt-in (requires an LLM call) so demo mode can produce a
full eval result for free.
"""

from __future__ import annotations

import json

from ptscribe import llm
from ptscribe.hallucination import find_hallucinations
from ptscribe.models import CheckStatus, EvalResult, SOAPNote


JUDGE_SYSTEM = """You evaluate whether a SOAP note's narrative sections faithfully capture a clinician's transcript.

You will receive: a transcript, and the narrative parts of the SOAP note (the patient history, the assessment summary, and the progress statement).

Score the narrative on three dimensions, each 0..1:
  - faithfulness: does the note misrepresent or invent clinical facts not in the transcript?
  - completeness: does it capture the key clinical points the clinician made?
  - clinical voice: does it read like a real PT/OT/ST clinician wrote it?

Return ONLY a JSON object: {"score": 0.0-1.0, "reasoning": "one sentence explaining the score"}

`score` is the mean of the three dimensions. Be honest — if the narrative is thin or wrong, score it low."""


# -- structure completeness --------------------------------------------------

# Each section's expected slots. A slot is "filled" when truthy (non-empty
# string, non-empty list, populated sub-model). 4 sections × N slots → score.
_EXPECTED_SLOTS = {
    "subjective": ("chief_complaint", "history", "pain", "patient_goals"),
    "objective": ("rom", "strength", "observations"),
    "assessment": ("summary", "progress"),
    "plan": ("interventions_today", "home_exercise_program", "next_visit"),
}


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def section_completeness(note: SOAPNote) -> tuple[float, bool]:
    """Return (completeness 0..1, has_all_four_sections)."""
    filled_total = 0
    expected_total = 0
    section_populated = []
    for section_name, slots in _EXPECTED_SLOTS.items():
        section = getattr(note, section_name)
        any_in_section = False
        for slot in slots:
            expected_total += 1
            value = getattr(section, slot, None)
            if _is_filled(value):
                filled_total += 1
                any_in_section = True
        section_populated.append(any_in_section)
    completeness = filled_total / expected_total if expected_total else 0.0
    return round(completeness, 3), all(section_populated)


# -- LLM-as-judge ------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def llm_as_judge(
    transcript: str,
    note: SOAPNote,
    model: str = "claude-sonnet-4-6",
) -> tuple[float | None, str, float]:
    """Score the note's narrative quality via a separate LLM call.

    Returns (score, reasoning, cost_usd). Score is None on parse failure.
    """
    narrative = (
        f"SUBJECTIVE HISTORY:\n{note.subjective.history.strip() or '(empty)'}\n\n"
        f"ASSESSMENT SUMMARY:\n{note.assessment.summary.strip() or '(empty)'}\n\n"
        f"PROGRESS:\n{note.assessment.progress.strip() or '(empty)'}"
    )
    user = (
        f"# Transcript\n{transcript.strip()}\n\n"
        f"# SOAP narrative sections\n{narrative}\n\n"
        "Score the narrative now."
    )
    result = llm.chat(
        model=model,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=200,
    )
    obj = _extract_json(result.text)
    score = obj.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    reasoning = str(obj.get("reasoning", "")).strip()
    return score_f, reasoning, result.cost_usd


# -- end-to-end --------------------------------------------------------------

def run_eval(
    transcript: str,
    note: SOAPNote,
    transcript_id: str = "",
    use_judge: bool = False,
    judge_model: str = "claude-sonnet-4-6",
) -> tuple[EvalResult, float]:
    """Run the full eval harness on one (transcript, SOAP) pair.

    Returns (EvalResult, additional_cost_usd_from_judge).
    """
    completeness, has_all = section_completeness(note)
    findings = find_hallucinations(transcript, note)
    judge_score: float | None = None
    judge_reasoning = ""
    judge_cost = 0.0
    if use_judge:
        judge_score, judge_reasoning, judge_cost = llm_as_judge(
            transcript, note, model=judge_model
        )

    # Overall status: any hallucination -> at least WARN. Big gaps -> FAIL.
    overall = CheckStatus.PASS
    if not has_all:
        overall = CheckStatus.FAIL
    if findings:
        # Treat high-confidence findings as FAIL, otherwise WARN.
        if any(f.confidence >= 0.8 for f in findings) and overall != CheckStatus.FAIL:
            overall = CheckStatus.WARN
        elif overall == CheckStatus.PASS:
            overall = CheckStatus.WARN
    if completeness < 0.4:
        overall = CheckStatus.FAIL
    if judge_score is not None and judge_score < 0.4:
        overall = CheckStatus.FAIL

    return EvalResult(
        transcript_id=transcript_id,
        has_all_sections=has_all,
        completeness_score=completeness,
        hallucination_findings=findings,
        judge_score=judge_score,
        judge_reasoning=judge_reasoning,
        overall=overall,
    ), judge_cost
