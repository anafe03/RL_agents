"""Scenario engine — the tick loop.

Each tick:
  1. Every agent observes the running shared transcript (what's been said so far).
  2. Every agent decides what to do this tick (speak / propose / decide / pass)
     by being asked their LLM, with their persona as the system prompt and
     the shared transcript as the conversation context.
  3. Their action is appended to the tick's actions list AND to the shared
     transcript view that the *next* speaker will see.

This is a deliberately simple model. No memory consolidation, no "reflect at
end of day" pass, no event injection in v0.1. Those land in v0.2 / v0.3.

Speakers per tick are interleaved (round-robin), but on every tick the order
rotates so no agent is permanently last.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from simulacrum import llm
from simulacrum.models import (
    Action,
    ActionType,
    AgentSpec,
    Scenario,
    Tick,
    Transcript,
)


def _system_for_agent(scenario: Scenario, agent: AgentSpec) -> str:
    return f"""# Who you are

{agent.persona}

# The setting

{scenario.setting.strip()}

# The shared goal of the room (everyone is loosely working toward this)

{scenario.shared_goal.strip()}

# Your private goal (do not say this out loud verbatim — let it shape what you push for)

{agent.private_goal}

# How to respond

You are one voice in a multi-person conversation. Each turn, respond with ONE message of dialogue — what you would actually say in this moment, in your own voice. Stay in character. Be specific. Don't summarize. Don't break the fourth wall. Don't narrate ("Alex thinks for a moment..."). Just speak.

Keep responses under 4 sentences unless you are making a concrete proposal that needs more.

If the conversation has reached a satisfying conclusion and there is nothing meaningful left for you to add this round, you may respond with exactly the word PASS (uppercase). Use this sparingly — only when staying silent is the right move.
""".strip()


def _format_transcript_so_far(actions: list[Action], speakers: dict[str, str]) -> str:
    if not actions:
        return "(no one has spoken yet — you're opening the conversation)"
    lines: list[str] = []
    for a in actions:
        name = speakers.get(a.actor_id, a.actor_id)
        lines.append(f"{name}: {a.content}")
    return "\n\n".join(lines)


def _next_speakers(scenario: Scenario, tick_number: int) -> Iterable[AgentSpec]:
    """Rotate the speaker order each tick so the same person isn't always last."""
    agents = list(scenario.agents)
    offset = tick_number % len(agents)
    return agents[offset:] + agents[:offset]


def run_scenario(scenario: Scenario, max_ticks: int | None = None) -> Transcript:
    """Drive the scenario through ticks and return the full transcript."""
    transcript = Transcript(scenario_name=scenario.name)
    speaker_names = {a.id: a.name for a in scenario.agents}
    all_actions: list[Action] = []  # rolling transcript visible to every agent
    n = max_ticks if max_ticks is not None else scenario.max_ticks

    for tick_number in range(1, n + 1):
        tick = Tick(number=tick_number)
        consecutive_passes = 0
        for agent in _next_speakers(scenario, tick_number):
            transcript_text = _format_transcript_so_far(all_actions, speaker_names)
            user_payload = (
                f"# Conversation so far (tick {tick_number} of {n})\n\n"
                f"{transcript_text}\n\n"
                f"# Your turn, {agent.name}. Respond now in character."
            )
            result = llm.chat(
                model=scenario.model,
                system=_system_for_agent(scenario, agent),
                messages=[{"role": "user", "content": user_payload}],
                max_tokens=400,
            )
            transcript.cost_usd += result.cost_usd
            content = result.text.strip()
            if content.upper() == "PASS":
                action = Action(type=ActionType.PASS, actor_id=agent.id, content="(passes)")
                consecutive_passes += 1
            else:
                action = Action(type=ActionType.SPEAK, actor_id=agent.id, content=content)
                consecutive_passes = 0
            tick.actions.append(action)
            all_actions.append(action)
        transcript.ticks.append(tick)
        # If everyone passed this tick, the conversation is genuinely done.
        if consecutive_passes == len(scenario.agents):
            break

    transcript.ended_at = datetime.now(timezone.utc)
    return transcript
