"""Deterministic mock chat function — produces a realistic trip-planning
transcript without calling the real Anthropic API.

Used by:
- `examples/demo_mock_trip_planning.py` (CLI demo)
- `src/simulacrum/ui/app.py` (Streamlit UI demo mode)
"""

from __future__ import annotations

from typing import Any

import anthropic

from simulacrum import llm


# Canned dialogue keyed by (agent_id, tick). The engine rotates speaker order
# per tick, so the order in each tick is computed at runtime.
TRIP_PLANNING_SCRIPT: dict[tuple[str, int], str] = {
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

STARTUP_CSUITE_SCRIPT: dict[tuple[str, int], str] = {
    ("priya", 1): "Alright. The board wants enterprise traction by Q3. I'm not going to pretend we have unlimited optionality — we need to pick one direction this morning and commit. Let me hear where everyone lands.",
    ("marcus", 1): "Before we pick a direction, I want it on the record that any pilot we sign means freezing platform work for the duration. We don't have the people to do both. So 'pick one direction' for me means 'pick one direction *and* protect the engineering roadmap from a single customer.'",
    ("lila", 1): "Pick whatever direction — but every plan on this whiteboard right now is understaffed by at least three roles. I'm not going to be the one explaining in 4 months why we burned out the team. We can't say yes to anything without a hiring plan attached.",
    ("devin", 1): "Numbers: $14M in the bank. Current burn is $850K/month. That's 16.5 months. Twelve new hires at $250K loaded is another $250K/month — drops us to 12 months. Before we pick a direction I want everyone to know what each plan does to the runway.",
    ("priya", 2): "OK Devin, that's fair. So let's frame the decision around runway. Marcus, give me your honest number: if we pick *one* enterprise pilot and ship it in 6 months, how much engineering capacity does that cost?",
    ("marcus", 2): "Half the team. Six months. And that's only if the customer commits to a tight scope. If they keep adding requirements, it's the whole team and 9 months — and we ship a thing that's bespoke to them and useful to no one else.",
    ("lila", 2): "Which means we also need a customer success person dedicated to that one logo. That's another hire, full-time, plus a solutions engineer. Two roles just to support one pilot.",
    ("devin", 2): "So a pilot scenario is: half the eng team locked up + two ops hires + 6-9 months of revenue we can't recognize. Call it $4M of cost, with maybe $500K of pilot revenue offsetting. Runway impact: roughly -3 months. Doable. Not free.",
    ("priya", 3): "Here's my pitch: we pick one warehouse logo, tight scope — pallet handling, full stop, no scope creep. Marcus locks the scope in the SOW. Lila, you get your CS hire and your SE. Devin, we model it explicitly as a runway-buying exercise — we are buying a logo at the cost of 3 months runway.",
    ("marcus", 3): "I'll sign that. *Only* if the SOW says 'pallet handling, single SKU, single site, no integration with the customer's WMS.' The moment that scope expands I want the right to escalate to this room.",
    ("lila", 3): "Workable. I want it on the record that we are hiring against this — CS lead by week 4, SE by week 6. If either slips, the pilot slips.",
    ("devin", 3): "OK. So the agreement is: one warehouse pilot, tight scope, eng-frozen for 6 months on platform work, two ops hires, runway target 13 months exit. I can model the board update around that.",
    ("priya", 4): "Done. Let's spend the rest of the hour on which logo. Costco came back warmer last week — I think they're the right opening shot.",
    ("marcus", 4): "PASS",
    ("lila", 4): "PASS",
    ("devin", 4): "PASS",
    ("priya", 5): "PASS",
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


# Persona-text fragments unique to each agent, used to classify which agent
# is being asked by sniffing the system prompt.
_AGENT_MARKERS = {
    "alex": "logistics company",
    "maya": "freelance food writer",
    "sam": "rather be outside",
    "priya": "second-time founder",
    "marcus": "ex-Boston Dynamics",
    "lila": "ex-Stripe ops lead",
    "devin": "ex-Bain",
}


def make_mock_chat(scenario_name: str = "trip_planning") -> Any:
    """Return a chat fn that emulates the given scenario.

    `scenario_name` selects which canned script to use.
    """
    script = (
        STARTUP_CSUITE_SCRIPT if scenario_name == "startup_csuite" else TRIP_PLANNING_SCRIPT
    )
    state = {"tick": 0}

    def fn(*, system: str, messages: list, **kwargs: Any) -> llm.ChatResult:
        agent_id = "unknown"
        for aid, marker in _AGENT_MARKERS.items():
            if marker in system:
                agent_id = aid
                break
        # Identify the tick by counting "tick N of" in the user message.
        user_text = ""
        if messages and isinstance(messages[0].get("content"), str):
            user_text = messages[0]["content"]
        tick = state["tick"] or 1
        for i in range(1, 13):
            if f"tick {i} of" in user_text:
                tick = i
                break
        state["tick"] = tick

        line = script.get((agent_id, tick), "...")
        return llm.ChatResult(
            text=line, usage=_usage(), cost_usd=0.0008, model="claude-sonnet-4-6"
        )

    return fn
