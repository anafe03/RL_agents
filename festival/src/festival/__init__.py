"""Festival Companion — personalized festival schedule planner."""

from festival.models import (
    Artist,
    DaySchedule,
    Festival,
    Schedule,
    Set,
    SetRecommendation,
    TasteProfile,
    load_lineup,
)
from festival.scheduler import build_schedule

__version__ = "0.0.1"

__all__ = [
    "Artist",
    "DaySchedule",
    "Festival",
    "Schedule",
    "Set",
    "SetRecommendation",
    "TasteProfile",
    "build_schedule",
    "load_lineup",
]
