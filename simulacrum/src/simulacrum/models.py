"""Core data models for Simulacrum.

A scenario is a folder on disk:

    scenarios/<name>/
    ├── scenario.yaml         ← cast, setting, goals, max_ticks
    └── personas/<id>.md      ← one persona file per cast member

`load_scenario(path)` reads the folder and returns a `Scenario` object.
The engine drives that scenario through ticks, producing a `Transcript`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:8]


class ActionType(str, Enum):
    SPEAK = "speak"
    THINK = "think"
    PROPOSE = "propose"
    DECIDE = "decide"
    PASS = "pass"


class AgentSpec(BaseModel):
    """A cast member: identity + persona + private goal."""

    id: str
    name: str
    role: str = ""
    persona: str = ""  # loaded from personas/<id>.md
    private_goal: str = ""
    starting_memories: list[str] = Field(default_factory=list)


class Action(BaseModel):
    """One thing an agent did during a tick."""

    id: str = Field(default_factory=_uid)
    type: ActionType = ActionType.SPEAK
    actor_id: str
    target_id: str | None = None  # for speak-to-someone-specific
    content: str = ""
    timestamp: datetime = Field(default_factory=_now)


class Event(BaseModel):
    """An external event injected into the simulation."""

    id: str = Field(default_factory=_uid)
    at_tick: int
    description: str
    visible_to: list[str] = Field(default_factory=list)  # empty = everyone


class Tick(BaseModel):
    """One round of the simulation. Each tick, every agent gets to act once."""

    number: int
    actions: list[Action] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)


class Scenario(BaseModel):
    """A loaded scenario: cast + setting + goals + tick budget."""

    name: str
    title: str = ""
    setting: str = ""
    shared_goal: str = ""  # what the cast collectively is trying to do
    agents: list[AgentSpec]
    events: list[Event] = Field(default_factory=list)
    max_ticks: int = 12
    model: str = "claude-sonnet-4-6"

    def agent(self, id: str) -> AgentSpec | None:
        for a in self.agents:
            if a.id == id:
                return a
        return None


class Transcript(BaseModel):
    """The full record of a simulated run."""

    id: str = Field(default_factory=_uid)
    scenario_name: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    ticks: list[Tick] = Field(default_factory=list)
    cost_usd: float = 0.0


def load_scenario(path: Path | str) -> Scenario:
    """Load a scenario from its directory."""
    path = Path(path)
    config_path = path / "scenario.yaml"
    personas_dir = path / "personas"

    if not config_path.exists():
        raise FileNotFoundError(f"missing scenario.yaml at {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}

    # Hydrate each agent's persona from personas/<id>.md
    agents = []
    for a_raw in raw.get("agents", []):
        agent = AgentSpec(**a_raw)
        persona_path = personas_dir / f"{agent.id}.md"
        if persona_path.exists():
            agent.persona = persona_path.read_text().strip()
        agents.append(agent)
    raw["agents"] = agents

    raw.setdefault("name", path.name)
    raw["agents"] = agents
    return Scenario(**raw)
