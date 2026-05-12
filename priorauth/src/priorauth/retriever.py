"""Retriever — picks the subset of guidelines relevant to a case.

For v0.1 this is LLM-judged: we hand the model the full corpus (~3 KB) plus
the denial + patient context, and ask which guideline IDs are relevant.
That's cheap with prompt caching and avoids the operational complexity of
an embedding store for a small corpus.

For v0.2 we'd swap this for a ChromaDB-backed vector retriever; the
interface stays the same so the drafter doesn't change.
"""

from __future__ import annotations

import json
import re

from priorauth import llm
from priorauth.models import Case, Guideline

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a clinical knowledge retriever for a prior-authorization appeal pipeline.

You will receive:
1. A patient context + a denial letter from an insurer.
2. A list of clinical guideline excerpts with stable IDs.

Your job: return ONLY the IDs of guidelines genuinely relevant to rebutting this specific denial for this specific patient. Be precise — irrelevant guidelines waste the drafter's attention and degrade the final appeal.

Selection criteria:
- A guideline is RELEVANT if its content materially supports a rebuttal to the denial reason given this patient's context.
- A guideline is NOT relevant just because it shares topic keywords. Quality > coverage.
- 2-4 guidelines is typical. Fewer is better than padding.

Respond ONLY in JSON: {"relevant_guideline_ids": ["id1", "id2", ...]}"""


def _format_corpus(corpus: list[Guideline]) -> str:
    lines: list[str] = []
    for g in corpus:
        lines.append(f"- id: {g.id}")
        lines.append(f"  source: {g.source}")
        lines.append(f"  topic: {g.topic}")
        lines.append(f"  excerpt: {g.excerpt.strip()}")
        lines.append("")
    return "\n".join(lines)


def _format_case(case: Case) -> str:
    p = case.patient
    return f"""Patient: {p.demographics}
Diagnoses:
  - {chr(10).join(['  - ' + d for d in p.diagnoses]).strip()}
Medications tried:
  - {chr(10).join(['  - ' + m for m in p.medications_tried]).strip()}
Contraindications: {', '.join(p.contraindications) if p.contraindications else '(none documented)'}
Red flags: {', '.join(p.red_flags) if p.red_flags else '(none)'}
Clinical history: {p.clinical_history.strip()}

Insurer denial:
  Payer: {case.denial.payer}
  Reason: {case.denial.denial_reason}
  Cited policy: {case.denial.cited_policy}
  Requested service: {case.requested_service}
"""


def retrieve_relevant(
    case: Case,
    corpus: list[Guideline],
    model: str = DEFAULT_MODEL,
) -> tuple[list[Guideline], float]:
    """Return the subset of `corpus` that is relevant to this case, plus cost."""
    user_payload = f"""# Case

{_format_case(case)}

# Guideline corpus

{_format_corpus(corpus)}

Return the IDs you select now."""

    result = llm.chat(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=800,
    )
    ids = _parse_ids(result.text)
    by_id = {g.id: g for g in corpus}
    selected = [by_id[i] for i in ids if i in by_id]
    return selected, result.cost_usd


def _parse_ids(raw: str) -> list[str]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    ids = obj.get("relevant_guideline_ids") or obj.get("ids") or []
    # Be permissive about format
    return [str(i).strip() for i in ids if isinstance(i, str)]
