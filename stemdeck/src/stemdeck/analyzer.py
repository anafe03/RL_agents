"""Musical analysis — key detection and energy estimation.

Key detection uses the Krumhansl-Schmuckler algorithm: build a 12-bin
pitch-class histogram from the MIDI notes, correlate it against the 24
major/minor key profiles, and take the best-correlating key. Pure music
theory — no ML, no audio, no model weights.
"""

from __future__ import annotations

from stemdeck.models import Channel, Section, Song

_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles — perceived stability of each scale degree,
# index 0 = the tonic.
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation of two equal-length sequences."""
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = (var_a * var_b) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom


def detect_key(midi_notes: list[int]) -> str:
    """Return the best-fitting key name, e.g. "A minor", or "" if no notes."""
    if not midi_notes:
        return ""
    histogram = [0.0] * 12
    for note in midi_notes:
        histogram[note % 12] += 1.0

    best_corr = -2.0
    best_key = ""
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            # Rotate the profile so its tonic weight lands at pitch class `tonic`.
            rotated = [profile[(i - tonic) % 12] for i in range(12)]
            corr = _pearson(histogram, rotated)
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_NOTE_NAMES[tonic]} {mode}"
    return best_key


def estimate_energy(song: Song) -> int:
    """Heuristic 1-10 energy estimate from arrangement density."""
    energy = 3
    energy += min(3, len(song.tracks) // 2)
    if Section.DROP in song.sections:
        energy += 2
    if song.track_for(Channel.KICK) is not None:
        energy += 1
    if song.bpm >= 126:
        energy += 1
    return max(1, min(10, energy))


def analyze(song: Song) -> Song:
    """Fill in `key` and `energy` for a song parsed from a `.als`.

    Returns a new `Song` — does not mutate the input. `camelot` is left for
    `compat.key_to_camelot` so the wheel table lives in one place.
    """
    from stemdeck.compat import key_to_camelot

    key = detect_key(song.harmonic_notes)
    updated = song.model_copy(update={"key": key, "camelot": key_to_camelot(key)})
    return updated.model_copy(update={"energy": estimate_energy(updated)})
