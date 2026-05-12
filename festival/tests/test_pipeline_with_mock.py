"""End-to-end mock pipeline test for Festival Companion.

Runs the matcher + scheduler against the bundled Skyline 2025 lineup with
the deterministic mock chat. Verifies the schedule has reasonable shape.
"""

from __future__ import annotations

from pathlib import Path

from festival import llm
from festival.matcher import score_lineup
from festival.mock import make_mock_chat
from festival.models import TasteProfile, load_lineup
from festival.scheduler import build_schedule


LINEUP_PATH = Path(__file__).resolve().parents[1] / "data" / "lineups" / "skyline_2025.yaml"


def test_full_pipeline_indie_listener():
    festival = load_lineup(LINEUP_PATH)
    profile = TasteProfile(
        description="indie rock, dream pop, sad indie folk; love Phoebe Bridgers, Beach House",
        favorite_artists=["Phoebe Bridgers", "Beach House"],
    )

    llm.set_chat_fn(make_mock_chat())
    try:
        recs, cost = score_lineup(festival, profile)
        schedule = build_schedule(festival, profile, recs)
    finally:
        llm.reset_chat_fn()

    assert len(schedule.days) == 3, "should have one DaySchedule per festival day"
    # All-non-empty taste should produce at least some picks
    assert schedule.total_picks > 0
    # Average score should be > 0
    assert schedule.average_score > 0
    # Indie-leaning user should have Phoebe Bridgers and Beach House in picks
    picked = {p.artist for d in schedule.days for p in d.picks}
    assert "Phoebe Bridgers" in picked, "indie listener should be assigned Phoebe Bridgers"
    assert "Beach House" in picked, "indie listener should be assigned Beach House"


def test_full_pipeline_electronic_listener():
    festival = load_lineup(LINEUP_PATH)
    profile = TasteProfile(
        description="electronic, dance, house, idm; love Caribou and Four Tet",
        favorite_artists=["Caribou", "Four Tet"],
    )

    llm.set_chat_fn(make_mock_chat())
    try:
        recs, cost = score_lineup(festival, profile)
        schedule = build_schedule(festival, profile, recs)
    finally:
        llm.reset_chat_fn()

    picked = {p.artist for d in schedule.days for p in d.picks}
    assert "Caribou" in picked
    assert "Four Tet" in picked


def test_must_see_overrides_low_score():
    festival = load_lineup(LINEUP_PATH)
    profile = TasteProfile(
        description="hip hop, rap; love Kendrick",
        favorite_artists=["Kendrick Lamar"],
        must_see=["Sufjan Stevens"],  # forced even though hip-hop fan wouldn't normally pick
    )

    llm.set_chat_fn(make_mock_chat())
    try:
        recs, cost = score_lineup(festival, profile)
        schedule = build_schedule(festival, profile, recs)
    finally:
        llm.reset_chat_fn()

    picked = {p.artist for d in schedule.days for p in d.picks}
    assert "Sufjan Stevens" in picked, "must-see artist must be locked in"


if __name__ == "__main__":
    test_full_pipeline_indie_listener()
    test_full_pipeline_electronic_listener()
    test_must_see_overrides_low_score()
    print("ok")
