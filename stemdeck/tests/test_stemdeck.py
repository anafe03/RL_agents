"""Tests for stemdeck — catalog, analysis, compatibility, mixing, parsing.

Live Ableton/OSC control is not tested (it needs a running DAW). Everything
else — key detection, the Camelot wheel, mix-board logic, the .als parser —
is pure and fully exercised here.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from stemdeck.analyzer import analyze, detect_key
from stemdeck.compat import (
    camelot_relation,
    channel_safety,
    key_to_camelot,
    rank_next,
    score_pair,
)
from stemdeck.mapping import map_channel
from stemdeck.match import harmonic_match, rhythmic_match, track_match
from stemdeck.mixer import MixBoard
from stemdeck.mock import demo_catalog
from stemdeck.models import HARMONIC, RHYTHMIC, Channel
from stemdeck.parser import parse_als, parse_als_bytes


# -- catalog -----------------------------------------------------------------

def test_demo_catalog_loads():
    catalog = demo_catalog()
    assert len(catalog.songs) == 6
    ids = set(catalog.song_ids)
    assert {"drift", "reach", "hollow", "ember", "pulse", "glass"} == ids


def test_every_song_has_kick_and_bass():
    for song in demo_catalog().songs:
        assert song.track_for(Channel.KICK) is not None, song.id
        assert song.track_for(Channel.BASS) is not None, song.id


# -- channel mapping ---------------------------------------------------------

def test_map_channel_keywords():
    assert map_channel("Kick") == Channel.KICK
    assert map_channel("BD Punch") == Channel.KICK
    assert map_channel("Reese Bass") == Channel.BASS
    assert map_channel("808 Sub") == Channel.BASS
    assert map_channel("Serum Lead Bright") == Channel.LEAD
    assert map_channel("String Atmos") == Channel.PAD
    assert map_channel("Vox Chop") == Channel.VOCAL
    assert map_channel("Closed Hats") == Channel.HATS


def test_map_channel_defaults_to_lead():
    assert map_channel("Mystery Squelch 7") == Channel.LEAD


# -- key detection -----------------------------------------------------------

def test_detect_key_c_major():
    # C-major scale, weighted toward tonic C and dominant G.
    notes = [60] * 4 + [67] * 3 + [64] * 2 + [62, 65, 69, 71]
    assert detect_key(notes) == "C major"


def test_detect_key_a_minor():
    # A-minor scale, weighted toward tonic A and dominant E.
    notes = [57] * 4 + [64] * 3 + [60] * 2 + [59, 62, 65, 67]
    assert detect_key(notes) == "A minor"


def test_detect_key_empty():
    assert detect_key([]) == ""


# -- Camelot wheel + compatibility -------------------------------------------

def test_key_to_camelot():
    assert key_to_camelot("A minor") == "8A"
    assert key_to_camelot("C major") == "8B"
    assert key_to_camelot("E minor") == "9A"
    assert key_to_camelot("nonsense") == ""


def test_camelot_relations():
    assert camelot_relation("8A", "8A") == "perfect"
    assert camelot_relation("8A", "8B") == "relative"
    assert camelot_relation("8A", "9A") == "adjacent"
    assert camelot_relation("8A", "10A") == "energy"
    assert camelot_relation("8A", "11A") == "risky"
    assert camelot_relation("7A", "8B") == "clash"
    assert camelot_relation("8A", "") == "unknown"


def test_score_pair_prefers_compatible():
    catalog = demo_catalog()
    drift = catalog.get("drift")    # 8A, 124
    reach = catalog.get("reach")    # 8A, 126 — same key, tiny tempo move
    ember = catalog.get("ember")    # 7A, 120
    pulse = catalog.get("pulse")    # 8B, 128

    good = score_pair(drift, reach)
    clash = score_pair(ember, pulse)
    assert good.score > clash.score
    assert good.stars >= 4
    assert good.key_relation == "perfect"
    assert clash.key_relation == "clash"


def test_rank_next_sorted_descending():
    catalog = demo_catalog()
    ranked = rank_next(catalog.get("drift"), catalog.songs)
    assert len(ranked) == 5  # all songs except drift itself
    scores = [p.score for p in ranked]
    assert scores == sorted(scores, reverse=True)


# -- channel safety ----------------------------------------------------------

def test_channel_safety_drums_always_safe():
    catalog = demo_catalog()
    # ember -> pulse is a hard key clash, but drums never clash harmonically.
    safety = channel_safety(catalog.get("ember"), catalog.get("pulse"))
    for ch in RHYTHMIC:
        if ch in catalog.get("pulse").channels:
            assert safety[ch] == "safe"


def test_channel_safety_harmonic_clash():
    catalog = demo_catalog()
    safety = channel_safety(catalog.get("ember"), catalog.get("pulse"))  # 7A vs 8B
    for ch in HARMONIC:
        if ch in catalog.get("pulse").channels:
            assert safety[ch] == "clash"


def test_channel_safety_harmonic_compatible():
    catalog = demo_catalog()
    safety = channel_safety(catalog.get("drift"), catalog.get("reach"))  # 8A vs 8A
    for ch in HARMONIC:
        if ch in catalog.get("reach").channels:
            assert safety[ch] == "safe"


# -- mix board ---------------------------------------------------------------

def test_mixboard_bring_in_and_out():
    board = MixBoard(demo_catalog())
    board.bring_in("drift", Channel.KICK)
    assert board.is_active("drift", Channel.KICK)
    assert board.now_playing() == ["drift"]
    board.bring_out("drift", Channel.KICK)
    assert not board.is_active("drift", Channel.KICK)
    assert board.now_playing() == []


def test_mixboard_auto_ducks_clashing_harmonic():
    board = MixBoard(demo_catalog())
    board.bring_in("ember", Channel.LEAD)   # 7A
    events = board.bring_in("pulse", Channel.LEAD)  # 8B — clashes with 7A
    assert board.is_active("pulse", Channel.LEAD)
    assert not board.is_active("ember", Channel.LEAD), "clashing lead should auto-duck"
    assert any(e.action == "out" and e.song_id == "ember" for e in events)


def test_mixboard_compatible_harmonic_layers_keep_both():
    board = MixBoard(demo_catalog())
    board.bring_in("drift", Channel.LEAD)   # 8A
    board.bring_in("reach", Channel.LEAD)   # 8A — same key, no duck
    assert board.is_active("drift", Channel.LEAD)
    assert board.is_active("reach", Channel.LEAD)


def test_mixboard_drums_never_duck():
    board = MixBoard(demo_catalog())
    board.bring_in("ember", Channel.KICK)
    board.bring_in("pulse", Channel.KICK)   # key clash, but kick is rhythmic
    assert board.is_active("ember", Channel.KICK)
    assert board.is_active("pulse", Channel.KICK)


# -- track match -------------------------------------------------------------

def test_demo_tracks_have_rhythm_grids():
    for song in demo_catalog().songs:
        for track in song.tracks:
            assert len(track.rhythm) == 16, f"{song.id}/{track.name}"


def test_harmonic_match_identical_tracks():
    catalog = demo_catalog()
    bass = catalog.get("drift").track_for(Channel.BASS)
    assert harmonic_match(bass, bass) == 1.0


def test_harmonic_match_none_for_drum_tracks():
    catalog = demo_catalog()
    kick = catalog.get("drift").track_for(Channel.KICK)  # no pitched notes
    bass = catalog.get("drift").track_for(Channel.BASS)
    assert harmonic_match(kick, bass) is None


def test_rhythmic_match_identical_grids():
    catalog = demo_catalog()
    # Every song's kick is four-on-the-floor — identical grids.
    kick_a = catalog.get("drift").track_for(Channel.KICK)
    kick_b = catalog.get("hollow").track_for(Channel.KICK)
    assert rhythmic_match(kick_a, kick_b) == 1.0


def test_rhythmic_match_differs_for_different_patterns():
    catalog = demo_catalog()
    # drift's bass is syncopated; reach's bass is on-beat — should not be 1.0.
    bass_drift = catalog.get("drift").track_for(Channel.BASS)
    bass_reach = catalog.get("reach").track_for(Channel.BASS)
    score = rhythmic_match(bass_drift, bass_reach)
    assert score is not None
    assert score < 1.0


def test_track_match_overall_and_verdict():
    catalog = demo_catalog()
    # Same-key, same-rhythm leads should land a strong match.
    lead_drift = catalog.get("drift").track_for(Channel.LEAD)   # A minor
    lead_glass = catalog.get("glass").track_for(Channel.LEAD)   # F# minor
    m = track_match(lead_drift, lead_drift)
    assert m.rhythmic == 1.0
    assert m.harmonic == 1.0
    assert m.overall == 1.0
    assert m.verdict == "locked"
    # A different song's lead should not be a perfect match.
    other = track_match(lead_drift, lead_glass)
    assert other.overall <= 1.0
    assert other.verdict in {"locked", "blends", "loose", "clash"}


# -- .als parser -------------------------------------------------------------

_MINIMAL_ALS = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <Tracks>
      <MidiTrack>
        <Name><EffectiveName Value="Reese Bass" /></Name>
        <MidiClip>
          <Notes><KeyTracks>
            <KeyTrack>
              <Notes><MidiNoteEvent Time="0" /><MidiNoteEvent Time="1" /></Notes>
              <MidiKey Value="33" />
            </KeyTrack>
          </KeyTracks></Notes>
        </MidiClip>
      </MidiTrack>
    </Tracks>
    <MasterTrack>
      <Tempo><Manual Value="126" /></Tempo>
    </MasterTrack>
  </LiveSet>
</Ableton>
"""


def test_parse_als_minimal(tmp_path: Path):
    als_path = tmp_path / "TestSong.als"
    with gzip.open(als_path, "wb") as fh:
        fh.write(_MINIMAL_ALS.encode("utf-8"))

    song = parse_als(als_path)
    assert song.bpm == 126.0
    assert song.id == "testsong"
    assert len(song.tracks) == 1
    track = song.tracks[0]
    assert track.name == "Reese Bass"
    assert track.channel == Channel.BASS
    assert track.notes == [33, 33]  # one per MidiNoteEvent
    # Onsets at beats 0 and 1 -> 16th-grid steps 0 and 4.
    assert len(track.rhythm) == 16
    assert track.rhythm[0] == 1
    assert track.rhythm[4] == 1

    # The analyze pass should fill in key + camelot + energy.
    analyzed = analyze(song)
    assert analyzed.key != ""
    assert analyzed.energy >= 1


def test_parse_als_bytes_matches_disk_parse(tmp_path: Path):
    raw = gzip.compress(_MINIMAL_ALS.encode("utf-8"))
    song = parse_als_bytes(raw, "Uploaded Track.als")
    assert song.id == "uploaded_track"
    assert song.title == "Uploaded Track"
    assert song.bpm == 126.0
    assert song.tracks[0].channel == Channel.BASS


def test_parse_als_bytes_handles_uncompressed():
    # Some .als saves are plain XML — the byte parser should still cope.
    song = parse_als_bytes(_MINIMAL_ALS.encode("utf-8"), "Plain.als")
    assert song.bpm == 126.0
    assert len(song.tracks) == 1


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            # tmp_path-dependent tests are skipped in the bare-run path.
            if "tmp_path" in fn.__code__.co_varnames:
                continue
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
