"""Extract a structured `Case` from a free-text insurance denial.

This is the consumer entry point: a patient pastes the denial letter they
received (plus, optionally, a few sentences about their situation), and an
LLM turns that loose text into the typed `Case` the appeal pipeline needs.

The LLM only *structures* what the letter says — it is instructed not to
invent clinical facts. Anything not stated comes back empty.
"""

from __future__ import annotations

import json
from uuid import uuid4

from priorauth import llm
from priorauth.models import Case, DenialLetter, PatientContext

INTAKE_SYSTEM = """You extract a structured prior-authorization case from a denial.

You receive an insurance denial letter, and optionally the patient's own
description of their situation. Return ONLY a JSON object, no prose:

{
  "title": "short case title",
  "requested_service": "the service, drug, or device that was denied",
  "patient": {
    "demographics": "e.g. 47-year-old female, or empty if not stated",
    "diagnoses": ["..."],
    "medications_tried": ["prior treatments/devices already tried"],
    "relevant_labs": {"name": "value"},
    "clinical_history": "a short free-text summary of the patient's situation",
    "contraindications": ["..."],
    "red_flags": ["urgent clinical findings, if any"]
  },
  "denial": {
    "payer": "the insurer's name",
    "member_id": "the member/subscriber id",
    "requested_service": "what was denied",
    "denial_reason": "the reason the insurer gave, in plain language",
    "cited_policy": "the policy name/section the denial cites, if any"
  }
}

Rules:
- Use "" for missing strings and [] for missing lists. Never guess.
- Do NOT invent clinical facts. Only structure what the letter and the
  patient description actually say.
- denial_reason should be a clear plain-language summary of why it was denied.
"""

_PATIENT_FIELDS = {
    "demographics", "diagnoses", "medications_tried", "relevant_labs",
    "clinical_history", "contraindications", "red_flags",
}
_DENIAL_FIELDS = {"payer", "member_id", "requested_service", "denial_reason", "cited_policy"}


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Intake LLM did not return JSON: {raw[:200]!r}") from None
        return json.loads(text[start : end + 1])


def coerce_case(raw_text: str, denial_letter: str) -> Case:
    """Turn the intake LLM's JSON reply into a validated `Case`."""
    obj = _extract_json(raw_text)
    case_id = f"pasted_{uuid4().hex[:8]}"

    patient_in = obj.get("patient") or {}
    patient_fields = {k: v for k, v in patient_in.items() if k in _PATIENT_FIELDS}

    denial_in = obj.get("denial") or {}
    denial_fields = {k: v for k, v in denial_in.items() if k in _DENIAL_FIELDS}

    return Case(
        id=case_id,
        title=str(obj.get("title") or "Pasted denial"),
        requested_service=str(obj.get("requested_service") or denial_fields.get("requested_service", "")),
        patient=PatientContext(case_id=case_id, **patient_fields),
        denial=DenialLetter(raw_text=denial_letter, **denial_fields),
    )


def extract_case(
    denial_letter: str,
    patient_situation: str = "",
    model: str = "claude-sonnet-4-6",
) -> tuple[Case, float]:
    """Extract a `Case` from a pasted denial letter. Returns (case, cost_usd)."""
    if not denial_letter.strip():
        raise ValueError("No denial letter text provided.")
    user_payload = (
        f"# Denial letter\n\n{denial_letter.strip()}\n\n"
        f"# Patient's description of their situation\n\n"
        f"{patient_situation.strip() or '(none provided)'}"
    )
    result = llm.chat(
        model=model,
        system=INTAKE_SYSTEM,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=900,
    )
    return coerce_case(result.text, denial_letter), result.cost_usd
