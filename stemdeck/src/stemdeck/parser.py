"""Parse Ableton `.als` project files.

A `.als` file is gzip-compressed XML. The schema shifts between Live
versions, so this parser is deliberately defensive: it pulls tempo, track
names, and MIDI note pitches where it can find them and tolerates missing
nodes rather than hard-failing. For anything it cannot read, the catalog
just carries less detail.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

from stemdeck.mapping import map_channel
from stemdeck.models import Section, Song, Track


def _read_als_xml(path: Path) -> str:
    """Return the XML text of a `.als` (gzipped) file."""
    with gzip.open(path, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


def _first_value(node: ET.Element, tag: str) -> str | None:
    """Find the first descendant `tag` and return its `Value` attribute."""
    found = node.find(f".//{tag}")
    if found is not None and "Value" in found.attrib:
        return found.attrib["Value"]
    return None


def _extract_tempo(root: ET.Element) -> float:
    """Tempo lives at MasterTrack > ... > Tempo > Manual[@Value]."""
    for tempo in root.iter("Tempo"):
        manual = tempo.find("Manual")
        if manual is not None and "Value" in manual.attrib:
            try:
                return float(manual.attrib["Value"])
            except ValueError:
                continue
    return 120.0


def _extract_track_name(track: ET.Element, fallback: str) -> str:
    name_node = track.find("Name")
    if name_node is not None:
        effective = name_node.find("EffectiveName")
        if effective is not None and effective.attrib.get("Value"):
            return effective.attrib["Value"]
        user = name_node.find("UserName")
        if user is not None and user.attrib.get("Value"):
            return user.attrib["Value"]
    return fallback


def _extract_midi_notes(track: ET.Element) -> list[int]:
    """Collect MIDI pitch numbers from every MIDI clip in the track.

    In the modern schema, notes sit under KeyTrack elements whose pitch is
    given by a sibling/child `MidiKey[@Value]`.
    """
    notes: list[int] = []
    for keytrack in track.iter("KeyTrack"):
        midi_key = keytrack.find("MidiKey")
        if midi_key is None or "Value" not in midi_key.attrib:
            continue
        try:
            pitch = int(midi_key.attrib["Value"])
        except ValueError:
            continue
        # One entry per note event at that pitch (weights the histogram).
        event_count = sum(1 for _ in keytrack.iter("MidiNoteEvent")) or 1
        notes.extend([pitch] * event_count)
    return notes


def _extract_rhythm(track: ET.Element) -> list[int]:
    """Collapse every MIDI note onset into one bar at 16th-note resolution.

    `MidiNoteEvent[@Time]` is the onset in beats; a 4/4 bar is 4 beats = 16
    sixteenths. Returns [] when there is no usable timing data.
    """
    grid = [0] * 16
    for event in track.iter("MidiNoteEvent"):
        time_str = event.attrib.get("Time")
        if time_str is None:
            continue
        try:
            beat = float(time_str)
        except ValueError:
            continue
        step = int((beat % 4.0) / 0.25) % 16
        grid[step] = 1
    return grid if any(grid) else []


def _count_clips(track: ET.Element) -> int:
    return sum(1 for _ in track.iter("MidiClip")) + sum(1 for _ in track.iter("AudioClip"))


def parse_als(path: Path | str) -> Song:
    """Parse a `.als` file into an un-analyzed `Song`.

    The returned song has tracks, tempo, and note data but no `key` or
    `energy` — run it through `analyzer.analyze` to fill those in.
    """
    path = Path(path)
    xml = _read_als_xml(path)
    root = ET.fromstring(xml)

    bpm = _extract_tempo(root)
    tracks: list[Track] = []
    for idx, track_tag in enumerate(("MidiTrack", "AudioTrack")):
        for node in root.iter(track_tag):
            is_midi = track_tag == "MidiTrack"
            name = _extract_track_name(node, fallback=f"{track_tag} {idx}")
            tracks.append(Track(
                name=name,
                channel=map_channel(name),
                is_midi=is_midi,
                notes=_extract_midi_notes(node) if is_midi else [],
                rhythm=_extract_rhythm(node) if is_midi else [],
                clip_count=_count_clips(node),
            ))

    return Song(
        id=path.stem.lower().replace(" ", "_"),
        title=path.stem,
        bpm=bpm,
        sections=list(Section),  # section detection is a separate pass; assume full arc
        tracks=tracks,
        source_path=str(path),
    )
