"""Poster → lineup: read a festival poster image with Claude vision.

The wedge, same as the scheduler: the LLM does the one thing only an LLM can do
(read a chaotic poster — arched type, wild fonts, tiered undercards — into
structured data), and deterministic code does the rest. Posters rarely print
set times, so `fill_missing_times()` synthesizes a plausible timetable in plain
Python: headliners close the night, tiers stack earlier, stages round-robin.
The weighted-interval scheduler downstream doesn't know or care that the slots
were synthesized.

Classic OCR is the wrong tool here — festival posters are adversarial
typography. A vision model reads them like a human does.
"""

from __future__ import annotations

import json
import os
import re

from festival.models import Artist, Festival, Set

VISION_MODEL = "claude-haiku-4-5"  # cheap + good at reading posters

_PROMPT = """Read this festival lineup poster and return STRICT JSON only (no prose, no code fences):

{
  "name": "<festival name, or 'Festival' if unreadable>",
  "city": "<city if shown, else ''>",
  "year": <year if shown, else 0>,
  "days": ["<day labels in order, e.g. 'Friday', 'Saturday' — [] if not shown>"],
  "artists": [
    {"name": "<artist>", "day": "<day label or ''>", "tier": <1 headliner / 2 support / 3 undercard>,
     "stage": "<stage if printed, else ''>", "start": "<HH:MM if printed, else ''>", "end": "<HH:MM if printed, else ''>"}
  ]
}

Rules: every readable artist name exactly once; tier from type size/position (biggest names = 1);
if the poster groups artists under day headings, set day accordingly; include times/stages ONLY if
actually printed. Return the JSON object and nothing else."""


def extract_lineup_from_image(image_bytes: bytes, media_type: str,
                              api_key: str | None = None) -> tuple[Festival, dict]:
    """One Claude vision call → a scoreable, schedulable Festival.

    Returns (festival, meta) where meta has token counts for the cost line.
    """
    import base64

    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=VISION_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.b64encode(image_bytes).decode()}},
            {"type": "text", "text": _PROMPT},
        ]}],
    )
    text = resp.content[0].text
    meta = {"model": VISION_MODEL,
            "in_tokens": resp.usage.input_tokens, "out_tokens": resp.usage.output_tokens}
    return parse_poster_json(text), meta


def parse_poster_json(text: str) -> Festival:
    """Parse the model's JSON into a Festival (pure function — unit-testable)."""
    raw = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(raw)
    days = [str(d) for d in (data.get("days") or [])]
    entries = data.get("artists") or []
    if not days:
        seen_days = [e.get("day") for e in entries if e.get("day")]
        days = list(dict.fromkeys(seen_days)) or ["Day 1"]

    sets: list[Set] = []
    artists: list[Artist] = []
    for e in entries:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        artists.append(Artist(name=name, blurb=""))
        sets.append(Set(
            artist=name,
            stage=(e.get("stage") or "").strip() or "TBD",
            day=(e.get("day") or "").strip() or days[0],
            start=e.get("start") or "00:00",
            end=e.get("end") or "00:00",
            headliner=int(e.get("tier") or 3) == 1,
        ))
    fest = Festival(name=data.get("name") or "Festival", city=data.get("city") or "",
                    year=int(data.get("year") or 0), days=days,
                    stages=[], sets=sets, artists=artists)
    fest = fill_missing_times(fest, tiers={s.artist: int(e.get("tier") or 3)
                                           for s, e in zip(sets, entries) if s.artist})
    fest.stages = sorted({s.stage for s in fest.sets})
    return fest


def fill_missing_times(fest: Festival, tiers: dict[str, int] | None = None) -> Festival:
    """Deterministically synthesize a timetable for sets the poster didn't time.

    Per day: undercard (tier 3) from 14:00, support (tier 2) evenings, headliners
    (tier 1) close from 21:30. Two default stages alternate so the scheduler has
    real conflicts to solve — that's the point of the product.
    """
    tiers = tiers or {}

    def _mm(h: int, m: int = 0) -> str:
        return f"{h:02d}:{m:02d}"

    for day in fest.days:
        day_sets = [s for s in fest.sets if s.day == day]
        untimed = [s for s in day_sets if s.start == "00:00" and s.end == "00:00"]
        if not untimed:
            continue
        untimed.sort(key=lambda s: (tiers.get(s.artist, 3), s.artist))  # headliners first
        headline_after = {"Main Stage": 21 * 60 + 30, "Horizon Stage": 21 * 60}  # closers start here
        slot = {"Main Stage": 14 * 60, "Horizon Stage": 14 * 60 + 30}
        heads = [s for s in untimed if tiers.get(s.artist, 3) == 1]
        rest = [s for s in untimed if tiers.get(s.artist, 3) != 1]
        stages = list(slot)
        for i, s in enumerate(sorted(rest, key=lambda s: -tiers.get(s.artist, 3))):
            stg = stages[i % 2]
            start = slot[stg]
            dur = 60 if tiers.get(s.artist, 3) == 2 else 45
            if s.stage == "TBD":
                s.stage = stg
            s.start, s.end = _mm(start // 60, start % 60), _mm((start + dur) // 60, (start + dur) % 60)
            slot[stg] = start + dur + 15
        for i, s in enumerate(heads):
            stg = stages[i % 2]
            start = max(slot[stg], headline_after[stg])
            if s.stage == "TBD":
                s.stage = stg
            s.start, s.end = _mm(start // 60, start % 60), _mm(min(start + 90, 1439) // 60, min(start + 90, 1439) % 60)
            slot[stg] = start + 105
    return fest
