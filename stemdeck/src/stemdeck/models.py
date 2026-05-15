"""Core data models for stemdeck.

A `Catalog` is a set of `Song`s. Each `Song` has `Track`s, and every track
is assigned a canonical `Channel` — the columns of the live mixer board.
The fixed channel set is what lets a "Lead" in one song line up with a
"Lead" in another, which is the whole point of the tool.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Channel(str, Enum):
    """Canonical instrument channels — the fixed columns of the mixer board."""

    KICK = "kick"
    SNARE = "snare"
    HATS = "hats"
    BASS = "bass"
    LEAD = "lead"
    PAD = "pad"
    VOCAL = "vocal"
    FX = "fx"


# Drums layer freely across songs — only BPM has to agree, never key.
RHYTHMIC: set[Channel] = {Channel.KICK, Channel.SNARE, Channel.HATS}
# Harmonic channels carry pitch — layering these across songs needs key
# compatibility or you get a clash.
HARMONIC: set[Channel] = {Channel.BASS, Channel.LEAD, Channel.PAD, Channel.VOCAL}

# Left-to-right order the mixer board renders its columns in.
CHANNEL_ORDER: list[Channel] = [
    Channel.KICK,
    Channel.SNARE,
    Channel.HATS,
    Channel.BASS,
    Channel.LEAD,
    Channel.PAD,
    Channel.VOCAL,
    Channel.FX,
]


class Section(str, Enum):
    """A song section — one scene-row's worth of a song."""

    INTRO = "intro"
    VERSE = "verse"
    BUILD = "build"
    CHORUS = "chorus"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"


class Track(BaseModel):
    """One instrument track lifted from a `.als` project."""

    name: str  # original track name from the Ableton project
    channel: Channel  # canonical channel this track was mapped to
    is_midi: bool = True
    notes: list[int] = Field(default_factory=list)  # MIDI pitch numbers (empty for audio)
    # One bar of note onsets at 16th-note resolution: 16 ints, 0 = no hit.
    # Drives rhythmic-match scoring; empty when no timing data is available.
    rhythm: list[int] = Field(default_factory=list)
    clip_count: int = 0


class Song(BaseModel):
    """A single Ableton project, analyzed."""

    id: str
    title: str
    bpm: float
    key: str = ""  # human key, e.g. "A minor" (empty until analyzed)
    camelot: str = ""  # Camelot-wheel code, e.g. "8A"
    energy: int = 5  # 1-10 subjective energy
    sections: list[Section] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    source_path: str = ""

    def track_for(self, channel: Channel) -> Track | None:
        for t in self.tracks:
            if t.channel == channel:
                return t
        return None

    @property
    def channels(self) -> set[Channel]:
        return {t.channel for t in self.tracks}

    @property
    def harmonic_notes(self) -> list[int]:
        """All MIDI pitches from harmonic tracks — the input to key detection."""
        out: list[int] = []
        for t in self.tracks:
            if t.channel in HARMONIC:
                out.extend(t.notes)
        return out


class Catalog(BaseModel):
    """A collection of analyzed songs."""

    songs: list[Song] = Field(default_factory=list)

    def get(self, song_id: str) -> Song | None:
        for s in self.songs:
            if s.id == song_id:
                return s
        return None

    @property
    def song_ids(self) -> list[str]:
        return [s.id for s in self.songs]


def load_song(path: Path | str) -> Song:
    """Load a pre-analyzed song from a JSON file (the demo-catalog format)."""
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text())
    raw.setdefault("id", path.stem)
    return Song(**raw)


def load_catalog(directory: Path | str) -> Catalog:
    """Load every `*.json` song under a directory into a `Catalog`."""
    directory = Path(directory)
    songs = [load_song(p) for p in sorted(directory.glob("*.json"))]
    return Catalog(songs=songs)
