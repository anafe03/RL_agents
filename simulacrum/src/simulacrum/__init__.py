"""Simulacrum — multi-agent scenario engine."""

from simulacrum.engine import run_scenario
from simulacrum.models import (
    Action,
    ActionType,
    AgentSpec,
    Event,
    Scenario,
    Tick,
    Transcript,
    load_scenario,
)

__version__ = "0.0.1"

__all__ = [
    "Action",
    "ActionType",
    "AgentSpec",
    "Event",
    "Scenario",
    "Tick",
    "Transcript",
    "load_scenario",
    "run_scenario",
]
