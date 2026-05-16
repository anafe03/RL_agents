"""Harmonic + tempo compatibility — the safety rails.

`compat` answers two questions:
  1. Song-to-song: how well does song B follow song A? (`score_pair`)
  2. Channel-to-channel: given A is playing, which of B's channels are
     safe to layer in right now? (`channel_safety`)

Key compatibility uses the Camelot wheel, the harmonic-mixing convention
DJs use: relative major/minor and ±1 around the wheel are smooth moves.
"""

from __future__ import annotations

from pydantic import BaseModel

from stemdeck.models import HARMONIC, RHYTHMIC, Channel, Song

# Camelot wheel — every enharmonic spelling detect_key might emit maps here.
_CAMELOT: dict[str, str] = {
    "Ab minor": "1A", "G# minor": "1A", "B major": "1B",
    "Eb minor": "2A", "D# minor": "2A", "Gb major": "2B", "F# major": "2B",
    "Bb minor": "3A", "A# minor": "3A", "Db major": "3B", "C# major": "3B",
    "F minor": "4A", "Ab major": "4B", "G# major": "4B",
    "C minor": "5A", "Eb major": "5B", "D# major": "5B",
    "G minor": "6A", "Bb major": "6B", "A# major": "6B",
    "D minor": "7A", "F major": "7B",
    "A minor": "8A", "C major": "8B",
    "E minor": "9A", "G major": "9B",
    "B minor": "10A", "D major": "10B",
    "F# minor": "11A", "Gb minor": "11A", "A major": "11B",
    "Db minor": "12A", "C# minor": "12A", "E major": "12B",
}


def key_to_camelot(key: str) -> str:
    """Map a human key name (e.g. "A minor") to its Camelot code (e.g. "8A")."""
    return _CAMELOT.get(key.strip(), "")


def _parse_camelot(code: str) -> tuple[int, str] | None:
    code = code.strip().upper()
    if len(code) < 2 or code[-1] not in ("A", "B"):
        return None
    try:
        return int(code[:-1]), code[-1]
    except ValueError:
        return None


def camelot_relation(code_a: str, code_b: str) -> str:
    """Classify the harmonic move A -> B.

    Returns one of: "perfect", "relative", "adjacent", "energy", "risky",
    "clash", or "unknown" (when either code is missing).
    """
    a = _parse_camelot(code_a)
    b = _parse_camelot(code_b)
    if a is None or b is None:
        return "unknown"
    num_a, let_a = a
    num_b, let_b = b
    if num_a == num_b and let_a == let_b:
        return "perfect"
    if num_a == num_b and let_a != let_b:
        return "relative"  # relative major/minor — always smooth
    # Wheel distance, wrapping 1..12.
    diff = min((num_a - num_b) % 12, (num_b - num_a) % 12)
    if let_a == let_b and diff == 1:
        return "adjacent"
    if let_a == let_b and diff == 2:
        return "energy"  # a +2 move — bigger lift, still usable
    if let_a == let_b and diff == 3:
        return "risky"
    return "clash"


# Higher = smoother. Used to weight the pair score.
_RELATION_SCORE: dict[str, float] = {
    "perfect": 1.0,
    "relative": 0.95,
    "adjacent": 0.88,
    "energy": 0.62,
    "risky": 0.40,
    "clash": 0.15,
    "unknown": 0.5,
}


def bpm_relation(bpm_a: float, bpm_b: float) -> str:
    """Classify the tempo gap: "tight", "close", "stretch", or "hard"."""
    delta = abs(bpm_a - bpm_b)
    if delta <= 2:
        return "tight"
    if delta <= 5:
        return "close"
    if delta <= 10:
        return "stretch"
    return "hard"


_BPM_SCORE: dict[str, float] = {"tight": 1.0, "close": 0.85, "stretch": 0.55, "hard": 0.2}


class PairScore(BaseModel):
    """How well song B follows song A."""

    song_a: str
    song_b: str
    score: float  # 0..1 overall
    stars: int  # 1..5, for quick display
    key_relation: str
    bpm_relation: str
    bpm_delta: float
    note: str  # one-line human summary


def _summary(key_rel: str, bpm_rel: str, energy_delta: int) -> str:
    key_phrase = {
        "perfect": "same key",
        "relative": "relative major/minor",
        "adjacent": "adjacent key",
        "energy": "+2 key lift",
        "risky": "distant key",
        "clash": "key clash",
        "unknown": "key unknown",
    }[key_rel]
    bpm_phrase = {
        "tight": "tempo locked",
        "close": "easy tempo move",
        "stretch": "noticeable tempo stretch",
        "hard": "hard tempo jump",
    }[bpm_rel]
    if energy_delta >= 2:
        arc = " — builds energy"
    elif energy_delta <= -2:
        arc = " — cools down"
    else:
        arc = ""
    return f"{key_phrase}, {bpm_phrase}{arc}"


def score_pair(song_a: Song, song_b: Song) -> PairScore:
    """Score the transition song_a -> song_b."""
    key_rel = camelot_relation(song_a.camelot, song_b.camelot)
    bpm_rel = bpm_relation(song_a.bpm, song_b.bpm)
    # Weighted: key matters a bit more than tempo for a smooth blend.
    raw = 0.55 * _RELATION_SCORE[key_rel] + 0.45 * _BPM_SCORE[bpm_rel]
    stars = max(1, min(5, round(raw * 5)))
    return PairScore(
        song_a=song_a.id,
        song_b=song_b.id,
        score=round(raw, 3),
        stars=stars,
        key_relation=key_rel,
        bpm_relation=bpm_rel,
        bpm_delta=round(abs(song_a.bpm - song_b.bpm), 1),
        note=_summary(key_rel, bpm_rel, song_b.energy - song_a.energy),
    )


def rank_next(current: Song, catalog_songs: list[Song]) -> list[PairScore]:
    """Rank every other song as a candidate to follow `current`, best first."""
    scores = [score_pair(current, s) for s in catalog_songs if s.id != current.id]
    return sorted(scores, key=lambda p: p.score, reverse=True)


def suggest_set_order(songs: list[Song]) -> list[Song]:
    """Greedy set order: open on the lowest-energy song, then repeatedly
    append the highest-scoring compatible song still unused.

    Not globally optimal — a greedy chain — but it gives a sensible
    build-from-calm running order and a usable energy arc to plan around.
    """
    if not songs:
        return []
    remaining = list(songs)
    current = min(remaining, key=lambda s: (s.energy, s.id))
    remaining.remove(current)
    order = [current]
    while remaining:
        nxt = max(remaining, key=lambda s: (score_pair(current, s).score, -ord(s.id[0])))
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def channel_safety(playing: Song, incoming: Song) -> dict[Channel, str]:
    """For each channel `incoming` has, is it safe to layer over `playing`?

    Returns "safe", "caution", or "clash" per channel:
      - rhythmic channels are always "safe" (BPM, not key, governs them)
      - harmonic channels follow the Camelot relation between the two songs
      - FX is always "safe"
    """
    key_rel = camelot_relation(playing.camelot, incoming.camelot)
    harmonic_verdict = {
        "perfect": "safe",
        "relative": "safe",
        "adjacent": "safe",
        "energy": "caution",
        "risky": "caution",
        "clash": "clash",
        "unknown": "caution",
    }[key_rel]
    out: dict[Channel, str] = {}
    for channel in incoming.channels:
        if channel in RHYTHMIC or channel == Channel.FX:
            out[channel] = "safe"
        elif channel in HARMONIC:
            out[channel] = harmonic_verdict
    return out
