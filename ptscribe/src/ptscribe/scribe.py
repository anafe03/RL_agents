"""The scribe pipeline — transcript -> structured SOAPNote.

One LLM call, one structured output. The eval harness and the
hallucination checker run *after* this — never inside it — so the
extraction itself stays a single well-defined unit.
"""

from __future__ import annotations

import json

from ptscribe import llm
from ptscribe.models import SOAPNote
from ptscribe.prompts import SCRIBE_SYSTEM


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Scribe LLM did not return JSON: {raw[:200]!r}") from None
        return json.loads(text[start : end + 1])


def _scrub(obj: dict) -> dict:
    """Drop sub-entries with missing required numeric fields.

    The prompt asks the LLM not to invent numbers, which sometimes leads
    it to emit a ROM/MMT/pain entry with ``"degrees": null`` (or grade /
    score = null) when the transcript named a joint without quoting a
    value. Strict Pydantic would reject the whole note; we'd rather drop
    the entry and continue — and the eval harness will record that the
    section came back lighter than expected.
    """
    obj = dict(obj or {})
    objective = obj.get("objective")
    if isinstance(objective, dict):
        if isinstance(objective.get("rom"), list):
            objective["rom"] = [
                r for r in objective["rom"]
                if isinstance(r, dict) and r.get("degrees") is not None
            ]
        if isinstance(objective.get("strength"), list):
            objective["strength"] = [
                m for m in objective["strength"]
                if isinstance(m, dict) and m.get("grade") is not None
            ]
    subjective = obj.get("subjective")
    if isinstance(subjective, dict) and isinstance(subjective.get("pain"), list):
        subjective["pain"] = [
            p for p in subjective["pain"]
            if isinstance(p, dict) and p.get("score") is not None
        ]
    return obj


def extract_soap(
    transcript: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[SOAPNote, llm.ChatResult]:
    """Run the LLM scribe pipeline on a transcript.

    Returns the validated SOAPNote and the raw ChatResult (so callers can
    record cost/latency/output in the run log).
    """
    if not transcript.strip():
        raise ValueError("Empty transcript.")
    result = llm.chat(
        model=model,
        system=SCRIBE_SYSTEM,
        messages=[{"role": "user", "content": transcript.strip()}],
        max_tokens=2000,
    )
    obj = _scrub(_extract_json(result.text))
    note = SOAPNote.model_validate(obj)
    return note, result
