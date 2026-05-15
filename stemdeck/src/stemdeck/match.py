"""Track-to-track matchness — how well do two individual tracks layer?

`compat` works at the song level (key, BPM). `match` works one level down:
given two specific tracks, how well do they fit together?

  - rhythmic match — do their note onsets land on the same grid steps?
    Compared as the cosine similarity of their 16-step onset patterns.
  - harmonic match — do their pitches agree? Compared as the cosine
    similarity of their 12-bin pitch-class histograms.

Either dimension is `None` when the data does not exist (a drum track has
no pitches; an audio track may have no onset grid). The overall score is
the mean of whichever dimensions are available.
"""

from __future__ import annotations

from pydantic import BaseModel

from stemdeck.models import Track


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length non-negative vectors, 0..1."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pitch_class_histogram(notes: list[int]) -> list[float]:
    hist = [0.0] * 12
    for note in notes:
        hist[note % 12] += 1.0
    return hist


def harmonic_match(track_a: Track, track_b: Track) -> float | None:
    """0..1 pitch agreement, or None if either track has no pitched notes."""
    if not track_a.notes or not track_b.notes:
        return None
    return _cosine(
        _pitch_class_histogram(track_a.notes),
        _pitch_class_histogram(track_b.notes),
    )


def rhythmic_match(track_a: Track, track_b: Track) -> float | None:
    """0..1 onset-grid agreement, or None if either track has no rhythm grid."""
    if not track_a.rhythm or not track_b.rhythm:
        return None
    width = max(len(track_a.rhythm), len(track_b.rhythm))
    grid_a = [float(x) for x in track_a.rhythm] + [0.0] * (width - len(track_a.rhythm))
    grid_b = [float(x) for x in track_b.rhythm] + [0.0] * (width - len(track_b.rhythm))
    return _cosine(grid_a, grid_b)


class MatchScore(BaseModel):
    """How well two tracks layer."""

    rhythmic: float | None  # 0..1, or None when no onset data
    harmonic: float | None  # 0..1, or None when no pitch data
    overall: float  # 0..1, mean of the available dimensions
    verdict: str  # "locked" | "blends" | "loose" | "clash"


def _verdict(overall: float) -> str:
    if overall >= 0.80:
        return "locked"
    if overall >= 0.55:
        return "blends"
    if overall >= 0.30:
        return "loose"
    return "clash"


def track_match(track_a: Track, track_b: Track) -> MatchScore:
    """Score how well two tracks fit together, rhythmically and harmonically."""
    rhythmic = rhythmic_match(track_a, track_b)
    harmonic = harmonic_match(track_a, track_b)
    available = [v for v in (rhythmic, harmonic) if v is not None]
    overall = sum(available) / len(available) if available else 0.0
    return MatchScore(
        rhythmic=None if rhythmic is None else round(rhythmic, 3),
        harmonic=None if harmonic is None else round(harmonic, 3),
        overall=round(overall, 3),
        verdict=_verdict(overall),
    )
