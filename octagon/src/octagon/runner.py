"""Attack runner — pits attacks against defenders and produces an AuditReport.

The conversation loop is a standard Anthropic tool-use loop:

    1. Send the attacker's turn(s) as user messages.
    2. Call the defender LLM with `system` + `tools`.
    3. If the response has tool_use blocks, run the defender's execute_tool()
       for each one, append the assistant message + tool_results, and loop.
    4. Stop on stop_reason == "end_turn", on max turns, or on error.
    5. Pass the full transcript to the judge.

For multi-turn attacks (attack.turns is a list), each attacker turn is sent
as the next user message *after* the defender has fully responded to the
prior one — so the attacker can adapt across turns.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from octagon import llm
from octagon.judge import judge_attack
from octagon.models import (
    Attack,
    AttackResult,
    AuditReport,
    ConversationTurn,
    Outcome,
    ToolCall,
)


MAX_DEFENDER_TURNS = 10  # safety cap so a stuck agent doesn't loop forever


def load_attacks(attacks_dir: Path | str = "attacks") -> list[Attack]:
    """Load every attack YAML under `attacks_dir/<category>/*.yaml`."""
    attacks_dir = Path(attacks_dir)
    out: list[Attack] = []
    for yaml_path in sorted(attacks_dir.rglob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        raw.setdefault("category", yaml_path.parent.name)
        out.append(Attack(**raw))
    return out


def _attacker_turns(attack: Attack) -> list[str]:
    if attack.is_multi_turn:
        return list(attack.turns)
    return [attack.prompt] if attack.prompt else []


def run_attack(defender, attack: Attack) -> AttackResult:
    """Run a single attack against a defender and capture the result."""
    started = time.perf_counter()

    transcript: list[ConversationTurn] = []
    tool_calls_made: list[ToolCall] = []
    messages: list[dict[str, Any]] = []
    total_cost = 0.0
    error: str | None = None

    try:
        for attacker_text in _attacker_turns(attack):
            messages.append({"role": "user", "content": attacker_text})
            transcript.append(ConversationTurn(role="attacker", content=attacker_text))

            # Defender responds — may take multiple internal turns for tool_use.
            for _ in range(MAX_DEFENDER_TURNS):
                result = llm.chat(
                    model=defender.model,
                    system=defender.system_prompt,
                    messages=messages,
                    tools=defender.tool_schemas,
                    max_tokens=1024,
                    thinking_mode="disabled",  # production defenders rarely use thinking
                )
                total_cost += result.cost_usd

                # Capture the assistant turn into the transcript.
                turn_text_parts: list[str] = []
                turn_tool_calls: list[ToolCall] = []
                for block in result.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        turn_text_parts.append(block.text)
                    elif block_type == "tool_use":
                        tc = ToolCall(
                            name=block.name,
                            args=dict(block.input) if block.input else {},
                        )
                        turn_tool_calls.append(tc)
                        tool_calls_made.append(tc)
                transcript.append(
                    ConversationTurn(
                        role="defender",
                        content="\n".join(turn_text_parts),
                        tool_calls=list(turn_tool_calls),
                    )
                )

                # Always append the assistant message back with the SDK content
                # objects so tool_use blocks survive into the next request.
                messages.append({"role": "assistant", "content": result.content})

                if result.stop_reason != "tool_use":
                    break  # defender done with this attacker turn

                # Execute each tool the defender called, append tool_results.
                tool_results: list[dict[str, Any]] = []
                for block in result.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    try:
                        tool_out = defender.execute_tool(block.name, dict(block.input))
                        # update our ToolCall record with the result
                        for tc in turn_tool_calls:
                            if tc.name == block.name and tc.result is None:
                                tc.result = tool_out
                                break
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(tool_out),
                            }
                        )
                        transcript.append(
                            ConversationTurn(
                                role="tool",
                                content=str(tool_out),
                                tool_calls=[
                                    ToolCall(
                                        name=block.name,
                                        args=dict(block.input),
                                        result=tool_out,
                                    )
                                ],
                            )
                        )
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"tool_error: {e!r}",
                                "is_error": True,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                # ran out of turns
                error = f"defender exceeded {MAX_DEFENDER_TURNS} internal turns"
                break

    except Exception as e:
        error = f"runner_error: {e!r}"

    outcome, reasoning, confidence, judge_cost = judge_attack(
        attack, transcript, tool_calls_made
    )
    total_cost += judge_cost

    return AttackResult(
        attack_id=attack.id,
        attack_category=attack.category,
        defender_name=getattr(defender, "name", "unknown"),
        transcript=transcript,
        outcome=Outcome.ERROR if error else outcome,
        judge_reasoning=error or reasoning,
        judge_confidence=0.0 if error else confidence,
        tools_called=[t.name for t in tool_calls_made],
        duration_ms=(time.perf_counter() - started) * 1000,
        cost_usd=total_cost,
    )


def run_audit(defender, attacks: list[Attack]) -> AuditReport:
    """Run a full audit: every attack against the defender."""
    report = AuditReport(defender_name=getattr(defender, "name", "unknown"))
    for attack in attacks:
        report.results.append(run_attack(defender, attack))
    report.ended_at = datetime.now(timezone.utc)
    return report
