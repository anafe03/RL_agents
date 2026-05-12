"""Assessor — independent run that grades the drafted appeal against a rubric.

The drafter is incentivized to be persuasive. The assessor is incentivized to
find weaknesses. Different LLM call, different system prompt, harder model.

Returns an `AppealAssessment` with a verdict + weak-points list. If `weak_points`
is non-empty, the UI surfaces them so a clinician can review before sending.
"""

from __future__ import annotations

import json

from priorauth import llm
from priorauth.models import (
    Appeal,
    AppealAssessment,
    Case,
    Guideline,
    RubricVerdict,
)

DEFAULT_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are an independent reviewer auditing a draft prior-authorization appeal letter.

You will receive:
1. The case (patient context + denial letter) the appeal is responding to.
2. The list of clinical guidelines the drafter was allowed to cite.
3. The drafted appeal as structured JSON.

Your job: assess whether the appeal is ready to send to the insurer or needs revision. You are not writing the appeal — you are catching weak points.

Score each of the four rubric items:

1. **Addressed all denial criteria** — Does the appeal directly rebut every reason the insurer gave for denial? If the denial cites a specific policy section, does the appeal address that section?
2. **All clinical claims cited** — Is every guideline-related claim in the appeal backed by a citation to one of the provided guideline IDs? Are any cited guideline IDs not in the provided list (i.e., fabricated)?
3. **Patient facts accurate** — Does the appeal accurately represent the patient context, or does it embellish, invent details, or misrepresent what the record supports?
4. **Has clear ask** — Does the appeal end with a direct, formal request to overturn the denial?

Overall verdict:
- "excellent" — all 4 boxes pass, no weak points
- "strong" — 3-4 boxes pass with only minor issues
- "moderate" — 2-3 boxes pass; substantive issues to fix
- "weak" — 0-2 boxes pass; should not be sent as written

List specific weak points in `weak_points`. Each weak point is one actionable sentence — what is the concrete problem to fix?

Respond ONLY with JSON:
{
  "verdict": "excellent" | "strong" | "moderate" | "weak",
  "addressed_all_denial_criteria": <bool>,
  "all_claims_cited": <bool>,
  "patient_facts_accurate": <bool>,
  "has_clear_ask": <bool>,
  "reasoning": "<2-3 sentence overall summary>",
  "weak_points": ["<actionable issue 1>", ...]
}"""


def _format_appeal(appeal: Appeal) -> str:
    lines = [
        f"OPENING: {appeal.opening}",
        "",
        "CLINICAL RATIONALE:",
    ]
    for i, p in enumerate(appeal.clinical_rationale, 1):
        lines.append(f"  {i}. {p}")
    lines.append("")
    lines.append("CITATIONS:")
    for c in appeal.citations:
        lines.append(f"  - claim: {c.claim}")
        lines.append(f"    guideline_id: {c.guideline_id}")
        lines.append(f"    quoted: {c.quoted_excerpt[:200]}")
    lines.append("")
    lines.append(f"CLOSING: {appeal.closing}")
    return "\n".join(lines)


def assess_appeal(
    case: Case,
    appeal: Appeal,
    guidelines: list[Guideline],
    model: str = DEFAULT_MODEL,
) -> AppealAssessment:
    """Run the assessor — returns an `AppealAssessment` with verdict + weak points."""
    guideline_ids = [g.id for g in guidelines]
    user_payload = f"""# Case

Patient: {case.patient.demographics}
Diagnoses: {', '.join(case.patient.diagnoses)}
Denial reason: {case.denial.denial_reason}
Requested service: {case.requested_service}

# Guidelines that were available (only these IDs are valid citations)

{', '.join(guideline_ids)}

# Drafted appeal

{_format_appeal(appeal)}

Assess the appeal now."""

    result = llm.chat(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=1024,
    )
    obj = _parse_assessment_json(result.text)
    verdict_str = obj.get("verdict", "moderate")
    try:
        verdict = RubricVerdict(verdict_str)
    except ValueError:
        verdict = RubricVerdict.MODERATE

    return AppealAssessment(
        appeal_id=appeal.id,
        verdict=verdict,
        addressed_all_denial_criteria=bool(obj.get("addressed_all_denial_criteria", False)),
        all_claims_cited=bool(obj.get("all_claims_cited", False)),
        patient_facts_accurate=bool(obj.get("patient_facts_accurate", False)),
        has_clear_ask=bool(obj.get("has_clear_ask", False)),
        reasoning=str(obj.get("reasoning", "")),
        weak_points=[str(w) for w in obj.get("weak_points", []) if w],
        cost_usd=result.cost_usd,
    )


def _parse_assessment_json(raw: str) -> dict:
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
