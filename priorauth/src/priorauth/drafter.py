"""Drafter — composes a cited appeal letter for a case using only retrieved guidelines.

The drafter is given the patient context, the denial, and ONLY the guidelines the
retriever selected. Its system prompt makes citation enforcement explicit: every
clinical claim must reference one of the provided guideline IDs. Claims without
a matching guideline are flagged by the assessor downstream.

Output is parsed JSON into the `Appeal` model.
"""

from __future__ import annotations

import json

from priorauth import llm
from priorauth.models import Appeal, Case, Citation, Guideline

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a clinical writer drafting a formal prior-authorization appeal letter.

The user will give you:
1. A patient context + a denial letter from an insurer.
2. A small set of clinical guideline excerpts (3-5), each with a stable ID and source.

Your job: produce a structured appeal that overturns the denial. The appeal must be:

- **Strictly factual** about the patient context — do not embellish, invent diagnoses or labs, or stretch what the record supports.
- **Strictly cited** for every clinical claim — every assertion of what guidelines say must reference one of the provided guideline IDs. If you cannot support a claim with one of the provided guidelines, do not make it.
- **Direct** in addressing the denial reason — do not write generic appeal language; rebut the specific cited policy with the specific clinical facts and guideline excerpts.
- **Professional in tone** — first-person from the prescribing clinician, formal but not florid.

Output shape (strict JSON):
{
  "opening": "<1-2 sentences naming the patient, service, and denial reason>",
  "clinical_rationale": [
    "<numbered point 1>",
    "<numbered point 2>",
    ...
  ],
  "citations": [
    {"claim": "<clinical assertion>", "guideline_id": "<id from the provided corpus>", "quoted_excerpt": "<the supporting fragment from that guideline, ≤ 280 chars>"},
    ...
  ],
  "closing": "<1-2 sentence formal ask: please overturn the denial>"
}

Rules:
- Every `clinical_rationale` point that makes a guideline-supported claim must have a matching `citations` entry.
- `guideline_id` values MUST appear in the list of guideline IDs the user provided. Do not invent IDs.
- `quoted_excerpt` must be a verbatim fragment from the corresponding guideline excerpt — copy text exactly.
- Respond with ONLY the JSON object."""


def _format_guidelines(guidelines: list[Guideline]) -> str:
    lines: list[str] = []
    for g in guidelines:
        lines.append(f"### {g.id}")
        lines.append(f"Source: {g.source}")
        lines.append(f"Topic: {g.topic}")
        lines.append(f"Excerpt: {g.excerpt.strip()}")
        lines.append("")
    return "\n".join(lines)


def _format_case_for_drafter(case: Case) -> str:
    p = case.patient
    return f"""## Patient
- Demographics: {p.demographics}
- Diagnoses:
{chr(10).join('  - ' + d for d in p.diagnoses)}
- Medications tried:
{chr(10).join('  - ' + m for m in p.medications_tried)}
- Labs: {', '.join(f"{k}: {v}" for k, v in p.relevant_labs.items()) or '(none documented)'}
- Contraindications: {', '.join(p.contraindications) if p.contraindications else '(none documented)'}
- Red flags: {', '.join(p.red_flags) if p.red_flags else '(none)'}
- Clinical history: {p.clinical_history.strip()}

## Denial
- Payer: {case.denial.payer}
- Member ID: {case.denial.member_id}
- Requested service: {case.requested_service}
- Denial reason: {case.denial.denial_reason}
- Cited policy: {case.denial.cited_policy}
- Letter text:
{case.denial.raw_text.strip()}
"""


def draft_appeal(
    case: Case,
    guidelines: list[Guideline],
    model: str = DEFAULT_MODEL,
) -> tuple[Appeal, float]:
    """Draft an appeal for a case using only the provided guideline subset."""
    guideline_ids = [g.id for g in guidelines]
    user_payload = f"""{_format_case_for_drafter(case)}

## Provided guidelines (only these IDs may be cited)

The following guideline IDs are the ONLY ones you may reference: {', '.join(guideline_ids)}

{_format_guidelines(guidelines)}

Draft the appeal now as JSON."""

    result = llm.chat(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=2048,
    )
    obj = _parse_appeal_json(result.text)

    citations = [
        Citation(
            claim=c.get("claim", ""),
            guideline_id=c.get("guideline_id", ""),
            quoted_excerpt=c.get("quoted_excerpt", ""),
        )
        for c in obj.get("citations", [])
    ]
    appeal = Appeal(
        case_id=case.id,
        payer=case.denial.payer,
        requested_service=case.requested_service,
        opening=obj.get("opening", ""),
        clinical_rationale=obj.get("clinical_rationale", []),
        citations=citations,
        closing=obj.get("closing", ""),
        drafter_model=model,
    )
    return appeal, result.cost_usd


def _parse_appeal_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
