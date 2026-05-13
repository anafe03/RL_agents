"""LLM-judged retriever — the existing approach, repackaged as a Retriever.

Hands the full corpus + the case to the LLM and asks which guideline IDs
are relevant. Zero-shot — no embedding step. Cost: one LLM call per
retrieval (Sonnet 4.6 by default).
"""

from __future__ import annotations

import json

from priorauth import llm
from priorauth.models import Case, Guideline
from priorauth.retrievers.base import Retriever

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


class LLMJudgedRetriever(Retriever):
    name = "llm_judged"
    category = "llm_judged"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self.model = model
        self._corpus: list[Guideline] = []

    def index(self, guidelines: list[Guideline]) -> None:
        # No persistent index — the LLM sees the corpus inline at retrieval time.
        self._corpus = list(guidelines)

    def retrieve(self, case: Case, k: int = 5) -> list[Guideline]:
        if not self._corpus:
            return []
        user_payload = f"""# Case

{self._format_case(case)}

# Guideline corpus

{self._format_corpus()}

Return the IDs you select now."""

        result = llm.chat(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
            max_tokens=800,
        )
        self.cost_usd += result.cost_usd
        ids = self._parse_ids(result.text)
        by_id = {g.id: g for g in self._corpus}
        selected = [by_id[i] for i in ids if i in by_id]
        return selected[:k]

    def _format_corpus(self) -> str:
        lines: list[str] = []
        for g in self._corpus:
            lines.append(f"- id: {g.id}")
            lines.append(f"  source: {g.source}")
            lines.append(f"  topic: {g.topic}")
            lines.append(f"  excerpt: {g.excerpt.strip()}")
            lines.append("")
        return "\n".join(lines)

    def _format_case(self, case: Case) -> str:
        p = case.patient
        return f"""Patient: {p.demographics}
Diagnoses:
{chr(10).join('  - ' + d for d in p.diagnoses)}
Medications tried:
{chr(10).join('  - ' + m for m in p.medications_tried)}
Contraindications: {', '.join(p.contraindications) if p.contraindications else '(none documented)'}
Red flags: {', '.join(p.red_flags) if p.red_flags else '(none)'}
Clinical history: {p.clinical_history.strip()}

Insurer denial:
  Payer: {case.denial.payer}
  Reason: {case.denial.denial_reason}
  Cited policy: {case.denial.cited_policy}
  Requested service: {case.requested_service}
"""

    def _parse_ids(self, raw: str) -> list[str]:
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
        return [str(i).strip() for i in ids if isinstance(i, str)]
