"""Core data models for Festival Companion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:8]


def _parse_hhmm(s: str) -> int:
    """Parse 'HH:MM' into minutes-since-midnight (0..1440)."""
    if isinstance(s, int):
        return s
    h, m = s.split(":")
    return int(h) * 60 + int(m)


class Artist(BaseModel):
    name: str
    genres: list[str] = Field(default_factory=list)
    blurb: str = ""  # short artist description the matcher feeds to the LLM


class Set(BaseModel):
    """One performance at a festival — artist × stage × time window."""

    id: str = Field(default_factory=_uid)
    artist: str  # matches Artist.name
    stage: str
    day: str
    start: str  # "HH:MM"
    end: str
    headliner: bool = False

    @property
    def start_min(self) -> int:
        return _parse_hhmm(self.start)

    @property
    def end_min(self) -> int:
        # If end < start, we crossed midnight — clamp to end-of-day for v0.1
        e = _parse_hhmm(self.end)
        return e if e > self.start_min else 24 * 60

    @property
    def duration_min(self) -> int:
        return self.end_min - self.start_min

    def overlaps(self, other: "Set") -> bool:
        if self.day != other.day:
            return False
        return not (self.end_min <= other.start_min or other.end_min <= self.start_min)


class Festival(BaseModel):
    name: str
    city: str = ""
    year: int = 0
    days: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    sets: list[Set]
    artists: list[Artist] = Field(default_factory=list)

    @field_validator("days", mode="before")
    @classmethod
    def _ensure_days_list(cls, v: Any) -> Any:
        return v or []

    def artist(self, name: str) -> Artist | None:
        for a in self.artists:
            if a.name == name:
                return a
        return None

    def sets_on(self, day: str) -> list[Set]:
        return [s for s in self.sets if s.day == day]


class TasteProfile(BaseModel):
    """A user's taste, expressed however they want to.

    For v0.1 this is paste-your-taste: a freeform description plus optional
    explicit signals. The matcher feeds it to the LLM as system context.
    """

    description: str = ""  # freeform — "I like dream pop, sad indie, some hip hop"
    favorite_artists: list[str] = Field(default_factory=list)
    favorite_genres: list[str] = Field(default_factory=list)
    must_see: list[str] = Field(default_factory=list)  # artist names — hard-locked picks
    avoid: list[str] = Field(default_factory=list)  # artist names — never pick


class SetRecommendation(BaseModel):
    """The agent's verdict for one set."""

    set_id: str
    artist: str
    day: str
    stage: str
    start: str
    end: str
    score: float = 0.0  # 0..1 taste-match
    reasoning: str = ""  # LLM annotation
    must_see: bool = False  # locked by TasteProfile.must_see


class DaySchedule(BaseModel):
    """The scheduler's picks for one day, plus the great-but-skipped ones."""

    day: str
    picks: list[SetRecommendation] = Field(default_factory=list)
    skipped_due_to_conflict: list[SetRecommendation] = Field(default_factory=list)


class Schedule(BaseModel):
    """The full plan across all days."""

    id: str = Field(default_factory=_uid)
    festival_name: str
    taste: TasteProfile
    days: list[DaySchedule] = Field(default_factory=list)
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)

    @property
    def total_picks(self) -> int:
        return sum(len(d.picks) for d in self.days)

    @property
    def average_score(self) -> float:
        all_picks = [p for d in self.days for p in d.picks]
        if not all_picks:
            return 0.0
        return sum(p.score for p in all_picks) / len(all_picks)


def load_lineup(path: Path | str) -> Festival:
    """Load a festival lineup from a YAML file."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    # Sets and artists may be defined together; coerce strings to objects.
    artists_raw = raw.get("artists", [])
    artists: list[Artist] = []
    for a in artists_raw:
        if isinstance(a, str):
            artists.append(Artist(name=a))
        else:
            artists.append(Artist(**a))
    raw["artists"] = artists
    raw["sets"] = [Set(**s) if isinstance(s, dict) else s for s in raw.get("sets", [])]
    return Festival(**raw)


def list_lineups(lineups_dir: Path | str = "data/lineups") -> list[tuple[str, Path]]:
    """Return (festival_name, path) for every lineup YAML on disk."""
    lineups_dir = Path(lineups_dir)
    out: list[tuple[str, Path]] = []
    if not lineups_dir.exists():
        return out
    for child in sorted(lineups_dir.glob("*.yaml")):
        try:
            f = load_lineup(child)
            out.append((f.name, child))
        except Exception:
            continue
    return out
