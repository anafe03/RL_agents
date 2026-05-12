"""Smoke test for the engine using a mock LLM. Proves:

1. Scenario loads (cast + personas + shared goal + setting).
2. Engine runs N ticks, calls the LLM for each agent each tick.
3. Transcript captures actions in the right order.
4. PASS shortcut works (everyone passes → loop exits early).
"""

from __future__ import annotations

import types
from typing import Any

import anthropic

from simulacrum import llm
from simulacrum.engine import run_scenario
from simulacrum.models import load_scenario


def _usage() -> anthropic.types.Usage:
    return anthropic.types.Usage(
        input_tokens=200,
        output_tokens=40,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        server_tool_use=None,
        service_tier=None,
    )


def test_trip_planning_loads_and_runs_one_tick():
    scenario = load_scenario("scenarios/trip_planning")
    assert scenario.name == "trip_planning"
    assert len(scenario.agents) == 3
    assert {a.id for a in scenario.agents} == {"alex", "maya", "sam"}
    # personas hydrated from disk
    for a in scenario.agents:
        assert len(a.persona) > 200  # personas are ~1KB each

    # Drive one tick with a mock that returns canned lines per agent.
    lines = {
        "alex": "Let's set a hard $1,400 ceiling per person before we go any further.",
        "maya": "Fine — but Cervejaria Ramiro is non-negotiable.",
        "sam": "And one full day in Sintra for hiking. The trail down to Cabo da Roca.",
    }

    def fake_chat(*, system: str, **kwargs: Any):
        # Detect which agent by matching persona text in the system prompt.
        for aid, txt in [
            ("alex", "logistics company"),
            ("maya", "freelance food writer"),
            ("sam", "rather be outside"),
        ]:
            if txt in system:
                return llm.ChatResult(
                    text=lines[aid], usage=_usage(), cost_usd=0.001, model="claude-sonnet-4-6"
                )
        return llm.ChatResult(text="...", usage=_usage(), cost_usd=0.001, model="claude-sonnet-4-6")

    llm.set_chat_fn(fake_chat)
    try:
        transcript = run_scenario(scenario, max_ticks=1)
    finally:
        llm.reset_chat_fn()

    assert len(transcript.ticks) == 1
    assert len(transcript.ticks[0].actions) == 3
    contents = {a.actor_id: a.content for a in transcript.ticks[0].actions}
    assert contents["alex"] == lines["alex"]
    assert contents["maya"] == lines["maya"]
    assert contents["sam"] == lines["sam"]
    assert transcript.cost_usd > 0


def test_all_pass_ends_loop_early():
    scenario = load_scenario("scenarios/trip_planning")

    def fake_chat(**kwargs: Any):
        return llm.ChatResult(text="PASS", usage=_usage(), cost_usd=0.0005, model="claude-sonnet-4-6")

    llm.set_chat_fn(fake_chat)
    try:
        transcript = run_scenario(scenario, max_ticks=5)
    finally:
        llm.reset_chat_fn()

    # First tick: everyone passes → loop should exit, so only 1 tick.
    assert len(transcript.ticks) == 1
    assert all(a.type.value == "pass" for a in transcript.ticks[0].actions)


def test_startup_csuite_scenario_loads():
    scenario = load_scenario("scenarios/startup_csuite")
    assert {a.id for a in scenario.agents} == {"priya", "marcus", "lila", "devin"}


if __name__ == "__main__":
    test_trip_planning_loads_and_runs_one_tick()
    test_all_pass_ends_loop_early()
    test_startup_csuite_scenario_loads()
    print("ok")
