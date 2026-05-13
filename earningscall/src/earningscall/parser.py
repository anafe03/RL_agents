"""Raw-transcript parsing.

Two paths in:
1. YAML with pre-structured `turns` — used for the bundled demo data.
2. Raw pasted text — used in the UI when someone drops in a transcript
   they copied from Seeking Alpha / Motley Fool / a prep doc. Best-effort
   regex extraction of speaker-prefixed lines.

Both produce the same `Transcript` model so the extractor doesn't care
which source you used.
"""

from __future__ import annotations

import re

from earningscall.models import SpeakerRole, SpeakerTurn, Transcript


# Common speaker-line patterns in transcripts:
#   "Operator: Welcome to..."
#   "Jane Smith - CEO: Thanks, operator..."
#   "Jane Smith, Chief Executive Officer: Thanks..."
#   "Bob Doe (Goldman Sachs): Hi Jane..."
_SPEAKER_LINE = re.compile(
    r"""
    ^                                         # start of line
    (?P<name>[A-Z][A-Za-z.\-' ]{1,60})         # speaker name
    (?:                                         # optional role/affiliation
        \s*[-,–]\s*(?P<role>[A-Z][A-Za-z &.\-/]{1,80})   #   - CEO, Chief...
        |
        \s*\((?P<aff>[^)]{1,80})\)              #   (Goldman Sachs)
    )?
    \s*:\s*                                     # colon delimiter
    (?P<text>.*)                                # the actual line text
    $
    """,
    re.VERBOSE,
)


def parse_raw(raw_text: str, company: str = "", period: str = "") -> Transcript:
    """Best-effort parse of a pasted transcript blob into structured turns."""
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    turns: list[SpeakerTurn] = []
    current: SpeakerTurn | None = None
    for ln in lines:
        if not ln.strip():
            continue
        m = _SPEAKER_LINE.match(ln)
        if m:
            # Flush prior turn
            if current is not None:
                turns.append(current)
            name = m.group("name").strip()
            role_raw = (m.group("role") or "").strip()
            aff = (m.group("aff") or "").strip()
            role = _classify_role(name, role_raw, aff)
            current = SpeakerTurn(
                turn_id=len(turns),
                speaker_name=name,
                speaker_role=role,
                affiliation=aff,
                text=m.group("text").strip(),
                is_question="?" in m.group("text"),
            )
        else:
            # Continuation of the current speaker's text
            if current is not None:
                current.text = (current.text + " " + ln.strip()).strip()
                if "?" in ln:
                    current.is_question = True
            # else: orphan line before any speaker — drop
    if current is not None:
        turns.append(current)

    return Transcript(
        id="pasted",
        company=company,
        period=period,
        turns=turns,
        raw_text=raw_text,
    )


def _classify_role(name: str, role_text: str, affiliation: str) -> SpeakerRole:
    """Infer a SpeakerRole from the parsed role hint and affiliation."""
    role_l = role_text.lower()
    aff_l = affiliation.lower()
    name_l = name.lower()
    if "operator" in name_l or "operator" in role_l:
        return SpeakerRole.OPERATOR
    if "ceo" in role_l or "chief executive" in role_l:
        return SpeakerRole.CEO
    if "cfo" in role_l or "chief financial" in role_l:
        return SpeakerRole.CFO
    # Investment bank / sell-side affiliations → analyst
    bank_markers = ("sachs", "morgan", "barclays", "evercore", "ubs", "citi", "credit suisse",
                    "rbc", "raymond james", "needham", "wedbush", "cowen", "jefferies", "td",
                    "deutsche", "wells fargo", "research", "capital", "securities")
    if any(b in aff_l for b in bank_markers) or any(b in role_l for b in bank_markers):
        return SpeakerRole.ANALYST
    if role_text:
        return SpeakerRole.OTHER_EXEC
    return SpeakerRole.UNKNOWN
