"""Poster → lineup: the pure parts (JSON parsing + timetable synthesis)."""

from __future__ import annotations

import json

from festival.vision import fill_missing_times, parse_poster_json

POSTER_JSON = json.dumps({
    "name": "Skyline", "city": "Chicago", "year": 2026,
    "days": ["Saturday", "Sunday"],
    "artists": [
        {"name": "The Waveforms", "day": "Saturday", "tier": 1},
        {"name": "Neon Palms", "day": "Saturday", "tier": 2},
        {"name": "Girl Cactus", "day": "Saturday", "tier": 3},
        {"name": "Static Bloom", "day": "Sunday", "tier": 1},
        {"name": "Moth Radio", "day": "Sunday", "tier": 3},
    ],
})


def test_parse_poster_json_builds_schedulable_festival():
    fest = parse_poster_json(POSTER_JSON)
    assert fest.name == "Skyline" and fest.days == ["Saturday", "Sunday"]
    assert len(fest.sets) == 5 and len(fest.artists) == 5
    # every set got a real synthesized time window
    for s in fest.sets:
        assert s.start != "00:00" or s.end != "00:00"
        assert s.end_min > s.start_min
    # headliners close the night
    sat = {s.artist: s for s in fest.sets_on("Saturday")}
    assert sat["The Waveforms"].start_min >= sat["Neon Palms"].start_min
    assert sat["The Waveforms"].headliner is True


def test_parse_strips_code_fences_and_defaults_days():
    fenced = "```json\n" + json.dumps(
        {"name": "X", "artists": [{"name": "Solo Act", "tier": 1}]}) + "\n```"
    fest = parse_poster_json(fenced)
    assert fest.days == ["Day 1"]
    assert fest.sets[0].day == "Day 1"


def test_printed_times_are_kept():
    data = json.dumps({"name": "T", "days": ["Fri"], "artists": [
        {"name": "A", "day": "Fri", "tier": 1, "stage": "Main", "start": "22:00", "end": "23:30"},
        {"name": "B", "day": "Fri", "tier": 3},
    ]})
    fest = parse_poster_json(data)
    a = next(s for s in fest.sets if s.artist == "A")
    assert (a.start, a.end, a.stage) == ("22:00", "23:30", "Main")   # untouched
    b = next(s for s in fest.sets if s.artist == "B")
    assert b.start != "00:00"                                        # synthesized
