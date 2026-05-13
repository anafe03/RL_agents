"""Octagon — adversarial audit + tournament platform for LLM agents."""

from octagon.models import (
    Attack,
    AttackCategory,
    AttackResult,
    AuditReport,
    ConversationTurn,
    Outcome,
    ToolCall,
)

__version__ = "0.0.1"

__all__ = [
    "Attack",
    "AttackCategory",
    "AttackResult",
    "AuditReport",
    "ConversationTurn",
    "Outcome",
    "ToolCall",
]
