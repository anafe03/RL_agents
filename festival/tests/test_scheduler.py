"""Deterministic tests for the weighted interval scheduler.

The scheduler is pure — no LLM, no I/O. We can fully exercise it with
synthetic inputs.
"""

from __future__ import annotations

from festival.models import Set, SetRecommendation, TasteProfile
from festival.scheduler import _weighted_interval_schedule, build_schedule
from festival.models import Festival, Artist


def test_scheduler_picks_two_non_overlapping_over_one_high_score():
    # Two short low-score sets that fit back-to-back vs. one high-score set
    # spanning both. The two-set combined weight should win.
    items = [
        (0, 60, 0.5),    # 0:00-1:00 weight 0.5
        (60, 120, 0.5),  # 1:00-2:00 weight 0.5
        (30, 90, 0.8),   # 0:30-1:30 weight 0.8 (overlaps both)
    ]
    chosen = _weighted_interval_schedule(items)
    chosen_weight = sum(items[i][2] for i in chosen)
    assert chosen_weight == 1.0
    # Should be the two non-overlapping ones, not the single high-score
    assert sorted(chosen) == [0, 1]


def test_scheduler_picks_single_high_when_it_dominates():
    items = [
        (0, 60, 0.1),
        (60, 120, 0.1),
        (30, 90, 0.9),
    ]
    chosen = _weighted_interval_schedule(items)
    assert chosen == [2]


def test_scheduler_handles_empty():
    assert _weighted_interval_schedule([]) == []


def test_scheduler_handles_all_non_overlapping():
    items = [
        (0, 60, 0.5),
        (60, 120, 0.7),
        (120, 180, 0.3),
    ]
    chosen = _weighted_interval_schedule(items)
    assert sorted(chosen) == [0, 1, 2]


def test_build_schedule_respects_must_see():
    artist_a = Artist(name="A", genres=["indie"])
    artist_b = Artist(name="B", genres=["indie"])
    sets = [
        Set(id="s1", artist="A", stage="X", day="Fri", start="10:00", end="11:00"),
        Set(id="s2", artist="B", stage="Y", day="Fri", start="10:15", end="11:15"),
    ]
    festival = Festival(name="Test", days=["Fri"], sets=sets, artists=[artist_a, artist_b])
    # Score B higher than A, but A is a must-see → A should win the conflict.
    recs = [
        SetRecommendation(set_id="s1", artist="A", day="Fri", stage="X", start="10:00", end="11:00", score=0.4),
        SetRecommendation(set_id="s2", artist="B", day="Fri", stage="Y", start="10:15", end="11:15", score=0.9),
    ]
    profile = TasteProfile(must_see=["A"])
    schedule = build_schedule(festival, profile, recs)
    picked_artists = [p.artist for p in schedule.days[0].picks]
    assert "A" in picked_artists
    assert "B" not in picked_artists


def test_build_schedule_drops_avoided_artists():
    sets = [
        Set(id="s1", artist="A", stage="X", day="Fri", start="10:00", end="11:00"),
        Set(id="s2", artist="B", stage="Y", day="Fri", start="12:00", end="13:00"),
    ]
    festival = Festival(name="Test", days=["Fri"], sets=sets)
    recs = [
        SetRecommendation(set_id="s1", artist="A", day="Fri", stage="X", start="10:00", end="11:00", score=0.9),
        SetRecommendation(set_id="s2", artist="B", day="Fri", stage="Y", start="12:00", end="13:00", score=0.4),
    ]
    profile = TasteProfile(avoid=["A"])
    schedule = build_schedule(festival, profile, recs)
    picked_artists = [p.artist for p in schedule.days[0].picks]
    assert picked_artists == ["B"]
