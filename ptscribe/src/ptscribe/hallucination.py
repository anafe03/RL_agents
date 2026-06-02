"""Cross-check extracted SOAP claims against the source transcript.

The principle: every concrete clinical datum the model put in the SOAP
note (a ROM degree, an MMT grade, a pain score, a prescribed exercise)
must be traceable to something the clinician actually said in the
transcript. The checker flags anything it can't verify.

False positives are tolerated — the goal is to surface SUSPICIOUS items
for human review, not to be perfect.
"""

from __future__ import annotations

import re

from ptscribe.models import HallucinationFinding, SOAPNote


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and treat hyphens/slashes as spaces.

    This keeps multi-word phrases like 'cat-cow' or 'L4/L5' matchable
    after `_any_significant_word` tokenizes claims into `[a-z]+` words.
    """
    text = text.lower().replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", text).strip()


def _has_number(text: str, value: float) -> bool:
    """Check that the integer or short-decimal form of `value` is in `text`."""
    candidates = {str(int(value)) if value == int(value) else f"{value:g}", f"{value:g}"}
    return any(re.search(rf"\b{re.escape(c)}\b", text) for c in candidates)


def _any_significant_word(text: str, phrase: str, min_len: int = 3) -> bool:
    """True if any non-stop word of `phrase` (len > min_len) appears in `text`."""
    words = [w for w in re.findall(r"[a-z]+", phrase.lower()) if len(w) > min_len]
    return any(w in text for w in words)


def find_hallucinations(transcript: str, note: SOAPNote) -> list[HallucinationFinding]:
    """Return one HallucinationFinding per claim that can't be verified."""
    t = _normalize(transcript)
    findings: list[HallucinationFinding] = []

    # --- ROM measurements ---------------------------------------------------
    for i, rom in enumerate(note.objective.rom):
        has_joint = _any_significant_word(t, rom.joint, min_len=2)
        has_num = _has_number(t, rom.degrees)
        if not (has_joint and has_num):
            missing = []
            if not has_joint:
                missing.append(f"joint '{rom.joint}'")
            if not has_num:
                missing.append(f"value {rom.degrees:g}°")
            findings.append(HallucinationFinding(
                field_path=f"objective.rom[{i}]",
                claim=f"{rom.side.value} {rom.joint} {rom.degrees:g}°",
                confidence=0.9 if not has_num else 0.55,
                note="not verifiable in transcript: " + ", ".join(missing),
            ))

    # --- Manual muscle tests ------------------------------------------------
    for i, mmt in enumerate(note.objective.strength):
        has_muscle = _any_significant_word(t, mmt.muscle_group, min_len=3)
        has_grade = _has_number(t, mmt.grade)
        if not (has_muscle and has_grade):
            findings.append(HallucinationFinding(
                field_path=f"objective.strength[{i}]",
                claim=f"{mmt.muscle_group} MMT {mmt.grade:g}/5",
                confidence=0.8 if not has_muscle else 0.55,
                note="muscle or grade not verifiable",
            ))

    # --- Pain scores --------------------------------------------------------
    for i, pain in enumerate(note.subjective.pain):
        has_loc = _any_significant_word(t, pain.location, min_len=3)
        has_score = _has_number(t, float(pain.score))
        if not (has_loc and has_score):
            findings.append(HallucinationFinding(
                field_path=f"subjective.pain[{i}]",
                claim=f"{pain.location} pain {pain.score}/10",
                confidence=0.75,
                note="pain location or score not verifiable",
            ))

    # --- Home exercise program ---------------------------------------------
    for i, ex in enumerate(note.plan.home_exercise_program):
        # Exercise names can be short (cat-cow, SLR, TKE, quad sets) so allow
        # 3-letter words. Also try a normalized full-phrase substring match
        # for compound names where individual words may be too generic.
        ex_norm = _normalize(ex.name)
        has_literal = ex_norm and ex_norm in t
        has_word = _any_significant_word(t, ex.name, min_len=2)
        if not (has_literal or has_word):
            findings.append(HallucinationFinding(
                field_path=f"plan.home_exercise_program[{i}]",
                claim=f"prescribed: {ex.name}",
                confidence=0.85,
                note="exercise name not present in transcript — possible fabrication",
            ))

    # --- Special tests ------------------------------------------------------
    for i, test in enumerate(note.objective.special_tests):
        if not _any_significant_word(t, test, min_len=4):
            findings.append(HallucinationFinding(
                field_path=f"objective.special_tests[{i}]",
                claim=f"test: {test}",
                confidence=0.7,
                note="special test name not present in transcript",
            ))

    return findings
