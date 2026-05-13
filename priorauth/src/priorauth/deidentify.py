"""De-identification — scrub PHI from a Case before sending to an LLM API.

This is a HIPAA-shaped feature, not a HIPAA-compliance guarantee. Compliance
requires the full stack (BAAs, encryption, audit, access controls); this
module addresses one piece: keeping PHI out of LLM provider logs.

Strategy:
1. **Identifier-driven scrubbing** for known fields. We know the patient's
   name (from demographics), the member ID, the claim number — replace
   those specifically with stable tokens. Much more reliable than generic
   NER for these specific strings.
2. **Regex passes** for common PHI patterns: emails, US phone numbers,
   SSNs, dates, addresses with street identifiers.

Both passes produce a `DeidMapping` recording every replacement, so the
output can be re-identified later if needed (e.g., for display to the
clinician but not for transit to the LLM).
"""

from __future__ import annotations

import copy
import re
from typing import Pattern

from pydantic import BaseModel, Field

from priorauth.models import Case, DenialLetter, PatientContext


# ---- mapping --------------------------------------------------------------


class DeidMapping(BaseModel):
    """Reverse-mapping from token → original value.

    Stored separately from the de-identified case so a UI can re-identify
    for human display while keeping the LLM call payload PHI-free.
    """

    tokens: dict[str, str] = Field(default_factory=dict)
    # category counters so we get stable [PATIENT_1], [PATIENT_2] tokens
    counters: dict[str, int] = Field(default_factory=dict)

    def add(self, category: str, original: str) -> str:
        """Get-or-create a token for this category+original pair."""
        # Reuse existing token if we've already mapped this original value
        for tok, orig in self.tokens.items():
            if orig == original and tok.startswith(f"[{category}_"):
                return tok
        self.counters[category] = self.counters.get(category, 0) + 1
        tok = f"[{category}_{self.counters[category]}]"
        self.tokens[tok] = original
        return tok

    def restore(self, text: str) -> str:
        """Replace tokens with original values in the given text."""
        out = text
        # Replace longest tokens first to avoid prefix collisions
        for tok in sorted(self.tokens, key=len, reverse=True):
            out = out.replace(tok, self.tokens[tok])
        return out


# ---- regex patterns -------------------------------------------------------

# Order matters: more specific patterns first so they don't get clobbered
# by greedier ones. Address last because it can overlap with shorter matches.
_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    (
        "DATE",
        re.compile(
            r"\b("
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"        # 01/14/2026 or 1-14-26
            r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"          # 2026-01-14
            r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}"  # March 14, 2026
            r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
            r")\b"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z0-9.\-' ]{1,40}\s+"
            r"(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|"
            r"Road|Rd\.?|Lane|Ln\.?|Court|Ct\.?|Way|Place|Pl\.?)"
            r"(?:[,\s]+[A-Z][A-Za-z]{1,20})?"
            r"(?:[,\s]+[A-Z]{2})?"
            r"(?:[,\s]+\d{5}(?:-\d{4})?)?",
        ),
    ),
]


def _apply_regex_patterns(text: str, mapping: DeidMapping) -> str:
    if not text:
        return text
    for category, pat in _PATTERNS:
        def _replace(m: re.Match) -> str:
            return mapping.add(category, m.group(0))
        text = pat.sub(_replace, text)
    return text


# ---- public API -----------------------------------------------------------


class Deidentifier:
    """Run identifier-driven + regex-driven scrubbing on a Case.

    Usage:
        deid = Deidentifier()
        clean_case, mapping = deid.deidentify_case(original_case)
        # ... send clean_case to LLM ...
        reidentified_appeal = mapping.restore(appeal.opening)
    """

    def __init__(self, known_names: list[str] | None = None) -> None:
        # If callers know specific names to scrub (e.g. patient name parsed
        # from a chart), they pass them in. The default empty list relies
        # on the regex pass + structured-field scrubbing.
        self._known_names = list(known_names or [])

    def deidentify_case(self, case: Case) -> tuple[Case, DeidMapping]:
        mapping = DeidMapping()
        clean = copy.deepcopy(case)

        # Collect structured-field originals → token assignments so we can
        # apply them to free-text fields as well (the denial.raw_text may
        # contain the member_id verbatim, etc.).
        structured_replacements: list[tuple[str, str]] = []

        if clean.denial.member_id:
            original = clean.denial.member_id
            tok = mapping.add("MEMBER_ID", original)
            clean.denial.member_id = tok
            structured_replacements.append((original, tok))

        if clean.denial.payer:
            # Insurer name is NOT PHI under HIPAA Safe Harbor but it can
            # narrow re-identification; scrub conservatively.
            original = clean.denial.payer
            tok = mapping.add("PAYER", original)
            clean.denial.payer = tok
            structured_replacements.append((original, tok))

        # Caller-supplied known names get their own pre-substitution
        for name in self._known_names:
            if not name:
                continue
            tok = mapping.add("PATIENT", name)
            structured_replacements.append((name, tok))

        # Apply ALL structured-field replacements to every free-text field
        # before the regex pass kicks in.
        def _apply_struct(text: str) -> str:
            for original, tok in structured_replacements:
                text = text.replace(original, tok)
            return text

        clean.patient.demographics = _apply_struct(clean.patient.demographics)
        clean.patient.clinical_history = _apply_struct(clean.patient.clinical_history)
        clean.patient.diagnoses = [_apply_struct(d) for d in clean.patient.diagnoses]
        clean.patient.medications_tried = [_apply_struct(m) for m in clean.patient.medications_tried]
        clean.denial.raw_text = _apply_struct(clean.denial.raw_text)
        clean.denial.denial_reason = _apply_struct(clean.denial.denial_reason)
        clean.denial.cited_policy = _apply_struct(clean.denial.cited_policy)
        clean.denial.requested_service = _apply_struct(clean.denial.requested_service)
        clean.requested_service = _apply_struct(clean.requested_service)

        # 2) Regex pass on every free-text field
        clean.patient.demographics = _apply_regex_patterns(clean.patient.demographics, mapping)
        clean.patient.clinical_history = _apply_regex_patterns(clean.patient.clinical_history, mapping)
        clean.patient.diagnoses = [_apply_regex_patterns(d, mapping) for d in clean.patient.diagnoses]
        clean.patient.medications_tried = [
            _apply_regex_patterns(m, mapping) for m in clean.patient.medications_tried
        ]
        clean.patient.contraindications = [
            _apply_regex_patterns(c, mapping) for c in clean.patient.contraindications
        ]
        clean.patient.red_flags = [_apply_regex_patterns(r, mapping) for r in clean.patient.red_flags]
        clean.patient.relevant_labs = {
            k: _apply_regex_patterns(v, mapping) for k, v in clean.patient.relevant_labs.items()
        }

        clean.denial.raw_text = _apply_regex_patterns(clean.denial.raw_text, mapping)
        clean.denial.denial_reason = _apply_regex_patterns(clean.denial.denial_reason, mapping)
        clean.denial.cited_policy = _apply_regex_patterns(clean.denial.cited_policy, mapping)
        clean.denial.requested_service = _apply_regex_patterns(clean.denial.requested_service, mapping)

        clean.requested_service = _apply_regex_patterns(clean.requested_service, mapping)

        return clean, mapping

    def deidentify_text(self, text: str, mapping: DeidMapping | None = None) -> tuple[str, DeidMapping]:
        """Scrub a free-text blob (e.g. a denial letter pasted raw)."""
        m = mapping or DeidMapping()
        cleaned = _apply_regex_patterns(text, m)
        return cleaned, m
