"""Deterministic mock run of the trip_planning scenario — captures a sample
transcript without calling the real Anthropic API.

Real usage:
    simulacrum run scenarios/trip_planning --transcript trip.md

This script lets you see what the output looks like before spending a cent.
"""

from __future__ import annotations

from typing import Any

import anthropic

from simulacrum import llm
from simulacrum.engine import run_scenario
from simulacrum.models import load_scenario
from simulacrum.render import render_markdown, render_terminal


# Canned dialogue keyed by (agent_id, tick). The engine rotates speaker order
# per tick, so the order in each tick is computed at runtime.
_SCRIPT: dict[tuple[str, int], str] = {
    ("alex", 1): "OK before we go anywhere — I have a $1,400-per-person ceiling. That's flights, lodging, food, transit, everything. I want to set that as a hard line before we start dreaming.",
    ("maya", 1): "Fine, but I'm telling you now: Cervejaria Ramiro is on the list. That's a $50 dinner with wine and it is the best $50 we'll spend on this trip.",
    ("sam", 1): "I'll agree to your $1,400 if I get one full day outside the city. Sintra, hiking the trail down to Cabo da Roca, picnic at the end. That's the one thing I need.",
    ("alex", 2): "OK let's actually map this out. 4 days. Two in Lisbon proper, one Sintra day for Sam, one... wildcard? I can plug that into the spreadsheet if we agree.",
    ("maya", 2): "Day 4 wildcard = Cascais. We do the train out, beach in the morning, seafood lunch at Mar do Inferno, then back to the city. That covers my Cascais seafood requirement.",
    ("sam", 2): "Cascais works for me — I can rent a bike along the coastal path while Maya is at lunch. We can converge for dinner back in Alfama.",
    ("alex", 3): "Trains to Sintra and Cascais are like €5 each way per person. That's basically free. The actual cost driver is the dinners. Maya, what's the damage on Ramiro plus Mar do Inferno?",
    ("maya", 3): "Ramiro is maybe $55 each with wine. Mar do Inferno is closer to $40 each, less wine because we're going back to the city. So $95 across two dinners. That leaves you plenty for cheaper meals the rest of the trip.",
    ("sam", 3): "Honestly I'd skip a sit-down dinner the Sintra day. Buy bread, cheese, port, eat it on the cliffs. Cuts another $30 off and it's a better night.",
    ("alex", 4): "OK I think we have a plan. Day 1: Lisbon arrival, Alfama walk, cheap dinner. Day 2: Belém + Time Out Market for lunch, Ramiro for dinner. Day 3: Sintra hike with picnic. Day 4: Cascais beach, Mar do Inferno lunch, last night dinner back in the city. Numbers say we're at $1,320 per person. Are we good?",
    ("maya", 4): "I'm good. Ramiro and Mar do Inferno both confirmed. I'll even let the picnic dinner happen.",
    ("sam", 4): "Locked in. I'll send the bike rental link tonight.",
    ("alex", 5): "PASS",
    ("maya", 5): "PASS",
    ("sam", 5): "PASS",
}


def _usage() -> anthropic.types.Usage:
    return anthropic.types.Usage(
        input_tokens=350,
        output_tokens=80,
        cache_creation_input_tokens=180,
        cache_read_input_tokens=170,
        server_tool_use=None,
        service_tier=None,
    )


def make_mock() -> Any:
    state = {"tick": 0, "spoken_this_tick": set()}

    def fn(*, system: str, messages: list, **kwargs: Any) -> llm.ChatResult:
        # Identify the agent from the persona text in the system prompt.
        agent_id = "unknown"
        for marker, aid in [
            ("logistics company", "alex"),
            ("freelance food writer", "maya"),
            ("rather be outside", "sam"),
        ]:
            if marker in system:
                agent_id = aid
                break
        # Identify the tick by counting "tick N of" in the user message.
        user_text = ""
        if messages and isinstance(messages[0].get("content"), str):
            user_text = messages[0]["content"]
        tick = state["tick"]
        for i in range(1, 13):
            if f"tick {i} of" in user_text:
                tick = i
                break
        if tick != state["tick"]:
            state["tick"] = tick
            state["spoken_this_tick"] = set()
        state["spoken_this_tick"].add(agent_id)

        line = _SCRIPT.get((agent_id, tick), "...")
        return llm.ChatResult(
            text=line, usage=_usage(), cost_usd=0.0008, model="claude-sonnet-4-6"
        )

    return fn


def main() -> None:
    llm.set_chat_fn(make_mock())
    try:
        scenario = load_scenario("scenarios/trip_planning")
        transcript = run_scenario(scenario, max_ticks=6)
        render_terminal(scenario, transcript)
        md = render_markdown(scenario, transcript)
        with open("examples/SAMPLE_TRIP_TRANSCRIPT.md", "w") as f:
            f.write(md)
        print(f"\nWrote examples/SAMPLE_TRIP_TRANSCRIPT.md ({len(transcript.ticks)} ticks)")
    finally:
        llm.reset_chat_fn()


if __name__ == "__main__":
    main()
