"""Core data models for Octagon."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return uuid4().hex[:12]


class AttackCategory(str, Enum):
    AUTHORITY = "authority_impersonation"
    EMERGENCY = "emergency_framing"
    PROMPT_INJECTION = "prompt_injection"
    ROLE_CONFUSION = "role_confusion"
    INDIRECT_INJECTION = "indirect_injection"
    TOOL_ABUSE = "tool_argument_abuse"
    SOCIAL_CHAIN = "social_engineering_chain"


class Outcome(str, Enum):
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class Attack(BaseModel):
    """A single adversarial test case loaded from a YAML file."""

    id: str
    category: AttackCategory
    name: str
    description: str = ""
    prompt: str = ""
    turns: list[str] = Field(default_factory=list)
    intent: str = ""
    success_criteria: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.turns)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class ConversationTurn(BaseModel):
    role: str  # "attacker" | "defender" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)


class AttackResult(BaseModel):
    id: str = Field(default_factory=_uid)
    attack_id: str
    attack_category: AttackCategory
    defender_name: str
    transcript: list[ConversationTurn] = Field(default_factory=list)
    outcome: Outcome = Outcome.AMBIGUOUS
    judge_reasoning: str = ""
    judge_confidence: float = 0.0
    tools_called: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    ran_at: datetime = Field(default_factory=_now)


class AuditReport(BaseModel):
    id: str = Field(default_factory=_uid)
    defender_name: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    results: list[AttackResult] = Field(default_factory=list)

    @property
    def total_attacks(self) -> int:
        return len(self.results)

    @property
    def total_blocked(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.BLOCKED)

    @property
    def total_succeeded(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.SUCCEEDED)

    @property
    def block_rate(self) -> float:
        return self.total_blocked / self.total_attacks if self.results else 0.0

    @property
    def by_category(self) -> dict[str, dict[str, int]]:
        """category -> {blocked, succeeded, ambiguous, error, total}"""
        out: dict[str, dict[str, int]] = {}
        for r in self.results:
            bucket = out.setdefault(
                r.attack_category.value,
                {"blocked": 0, "succeeded": 0, "ambiguous": 0, "error": 0, "total": 0},
            )
            bucket[r.outcome.value] += 1
            bucket["total"] += 1
        return out

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)
