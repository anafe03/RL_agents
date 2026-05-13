"""InternalIT defender — corporate help desk bot with security rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from octagon.defenders.internal_it import tools

_HERE = Path(__file__).parent
_SYSTEM_PROMPT = (_HERE / "system_prompt.md").read_text().strip()


class InternalITDefender:
    """The InternalIT agent — system prompt + tool schemas + tool dispatcher.

    The actual LLM call lives in `octagon.runner` so that the defender stays
    a simple data-and-dispatch object. The runner takes this defender, sends
    messages to the LLM with `system_prompt` + `tool_schemas`, and calls
    `execute_tool()` to run any tools the LLM picks.
    """

    name: str = "internal_it"
    display_name: str = "InternalIT (Acme Corp help desk)"
    system_prompt: str = _SYSTEM_PROMPT
    tool_schemas: list[dict[str, Any]] = tools.TOOL_SCHEMAS

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "lookup_user":
            return tools.lookup_user(args["email"])
        if name == "check_open_tickets":
            return tools.check_open_tickets(args["user_id"])
        if name == "reset_password":
            return tools.reset_password(args["user_id"], args["verification_method"])
        return {"error": "unknown_tool", "name": name}
