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
    ("marcus", 1): "Before we pick a direction, I want it on the record that any pilot we sign means freezing platform work for the duration. We don't have the people to do both.",
    ("lila", 1): "Pick whatever direction — but every plan on this whiteboard right now is understaffed by at least three roles. We can't say yes to anything without a hiring plan attached.",
    ("devin", 1): "Numbers: $14M in the bank. Current burn $850K/month. That's 16.5 months. Twelve new hires drops us to 12. Everyone needs to know what each plan does to the runway.",
    ("priya", 2): "OK Devin, fair. Let's frame around runway. Marcus, if we pick *one* enterprise pilot and ship it in 6 months, how much engineering capacity does that cost?",
    ("marcus", 2): "Half the team. Six months. And only if the customer commits to a tight scope. If they keep adding requirements, it's the whole team and 9 months — and we ship a thing useful to no one else.",
    ("lila", 2): "Which means a dedicated CS hire and a solutions engineer for that one logo. Two roles just to support one pilot.",
    ("devin", 2): "So: half eng locked up + two ops hires + 6-9 months of no recognized revenue. Call it $4M cost, $500K pilot revenue offsetting. Runway impact: roughly -3 months. Doable. Not free.",
    ("priya", 3): "My pitch: one warehouse logo, tight scope — pallet handling, full stop, no scope creep. Marcus locks scope in the SOW. Lila gets her CS hire and SE. Devin, we model it as a runway-buying exercise.",
    ("marcus", 3): "I'll sign that. *Only* if the SOW says 'pallet handling, single SKU, single site, no integration with the customer's WMS.' The moment scope expands I want the right to escalate.",
    ("lila", 3): "Workable. CS lead by week 4, SE by week 6. If either slips, the pilot slips.",
    ("devin", 3): "Agreement: one warehouse pilot, tight scope, eng-frozen for 6 months on platform work, two ops hires, runway target 13 months exit. I'll model the board update around that.",
    ("priya", 4): "Done. Rest of the hour on which logo — Costco came back warmer last week.",
    ("marcus", 4): "PASS",
    ("lila", 4): "PASS",
    ("devin", 4): "PASS",
    ("priya", 5): "PASS",
}


VAN_LIFE_DETOUR_SCRIPT: dict[tuple[str, int], str] = {
    ("maddy", 1): "OK. Highway south is closed, no estimated reopen. Three real options I can see: east into the Wallowas, north into the Methow Valley, or west to the coast. Pick a direction in the next twenty minutes and we drive.",
    ("theo", 1): "If we're voting on aesthetics — the Wallowas are getting the wildfire light. Smoke-haze sunsets over basalt is the photo of the year. I'm pitching east.",
    ("reyes", 1): "Before we vote on vibes: we have half a tank, one full propane, one half. The Wallowas resupply town is Joseph, which closes at six. We won't make it. The Methow has Twisp, open until eight. Coast has everything.",
    ("maddy", 2): "Reyes, fair. So eastern Wallowas requires us to either resupply in Yakima before we leave or bail on hot dinners the next two days.",
    ("theo", 2): "Yakima route adds maybe an hour. Sunset's at 7:42, we still hit golden hour at Hat Point if we're moving by 1pm.",
    ("reyes", 2): "Yakima route I can live with. I'd rather drive ninety minutes more and eat real food than skip a meal for a photo we hope works out.",
    ("maddy", 3): "OK. Vote — Wallowas via Yakima resupply. We're moving in fifteen. Theo, you have until then to confirm Hat Point campground has space and the road in isn't itself on fire.",
    ("theo", 3): "Already pulling it up. One bar though — might need to drive ten minutes to check.",
    ("reyes", 3): "I'll do propane and water while you check. Maddy, can you map the route and tell me how much gas we burn before we'd be in Joseph tomorrow morning?",
    ("maddy", 4): "On it. We're good. Twenty minutes and we're rolling.",
    ("theo", 4): "PASS",
    ("reyes", 4): "PASS",
    ("maddy", 5): "PASS",
}


HEIST_CREW_SCRIPT: dict[tuple[str, int], str] = {
    ("nora", 1): "Last meeting. We go tomorrow night at 11:14pm, six minutes after the guard rotation hits the south gallery. Everyone tell me one thing they want resolved before we walk out of this room. Dane first.",
    ("dane", 1): "The alarm spec. Jules, you gave us a Bosch 9000 panel and cellular failover. I checked the manufacturer's update log. Bosch pushed a firmware revision three weeks ago that adds a tamper signal we didn't account for. Either the panel got updated or it didn't. I need to know.",
    ("rae", 1): "Three traffic cameras went up between June 1 and June 30 on the exit route. Beale and Carlton, Mission and 4th, Bryant and 7th. The route as drawn puts us through all three at 11:23, 11:25, 11:27pm. Either the cameras are off after midnight or we need a different exit.",
    ("jules", 1): "Bosch — the panel was updated. I watched the tech do it from the kitchen window. But the museum doesn't enable tamper monitoring; they never have. It's a configuration setting, not a firmware default. I can check it tomorrow morning if you want documentation.",
    ("nora", 2): "Documentation tomorrow morning works. Dane, does that satisfy you?",
    ("dane", 2): "If Jules can photograph the config screen and I can see 'tamper: disabled' in plain text, yes. Verbal confirmation, no.",
    ("jules", 2): "I can do that.",
    ("rae", 2): "Cameras. I have two routes: original through the three new arrays, or south via the bridge approach — adds 90 seconds, zero new cameras. The trade is the bridge approach has a single chokepoint if we're tailed.",
    ("nora", 3): "Take the bridge approach. Ninety seconds is acceptable. A chokepoint we can plan around; cameras we cannot. Anyone disagree?",
    ("dane", 3): "Bridge is fine.",
    ("jules", 3): "Fine.",
    ("rae", 3): "Done.",
    ("nora", 4): "Then tomorrow night, 11:14. Final check-in at 8pm; if anyone wants to scrub, you tell me by 8 and I scrub it.",
    ("dane", 4): "PASS",
    ("jules", 4): "PASS",
    ("rae", 4): "PASS",
    ("nora", 5): "PASS",
}


FAMILY_THANKSGIVING_SCRIPT: dict[tuple[str, int], str] = {
    ("linda", 1): "Mara, please — eat. There's so much, and you haven't tried the stuffing. Sasha, did you tell her about the stuffing? It's my mother's recipe.",
    ("bob", 1): "It's a good stuffing. Even when I burn it. Which I did once. Twenty-two years ago and she still brings it up.",
    ("sasha", 1): "Mom, she's eating. Mara, the stuffing is genuinely amazing — it has a thing with sage and apple, you'll like it.",
    ("jordan", 1): "Mara, Sasha said you teach at Northwestern? That must be intense this time of year — end of semester?",
    ("mara", 1): "Yeah, papers are coming in this week. The stuffing is great, Linda — is the apple Granny Smith?",
    ("linda", 2): "Honeycrisp, actually. Granny Smith gets watery. Honeycrisp holds its shape and you get a little sweetness against the sage.",
    ("bob", 2): "She has a *system*. Don't get her started.",
    ("sasha", 2): "Dad, let her be excited. She gets to be excited about the stuffing once a year, that's the deal.",
    ("jordan", 2): "Speaking of systems — did Mom tell you Mark and I picked the venue? It's this barn out in McHenry County, kind of run-down but in a charming way. We're getting it for, like, half what the downtown spots wanted.",
    ("mara", 2): "Half is excellent. Is that the one with the wraparound porch in the listing photos?",
    ("linda", 3): "Wait, you saw the photos? Sasha showed you the photos?",
    ("sasha", 3): "Mara was around when I was looking at them. We were on the couch. Mom, it's not a thing.",
    ("bob", 3): "Linda, eat your potatoes. The girls are getting along. Let it happen.",
    ("jordan", 3): "Mara — I have to ask, because Sasha will not — what's the dynamic in your family at Thanksgiving like? Is it like this?",
    ("mara", 3): "Honestly? Quieter. My parents and my brother and a lot of long pauses. I'm enjoying the noise.",
    ("linda", 4): "Oh — well, we can do quiet. We don't have to be loud. Were we being too loud?",
    ("sasha", 4): "Mom. She said she was enjoying it. Take the compliment.",
    ("bob", 4): "She's taking the compliment. She's just taking it with extra steps.",
    ("jordan", 4): "Mara, more wine?",
    ("mara", 4): "Yes please.",
    ("linda", 5): "PASS",
    ("bob", 5): "PASS",
    ("sasha", 5): "PASS",
    ("jordan", 5): "PASS",
    ("mara", 5): "PASS",
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
    # trip_planning
    "alex": "logistics company",
    "maya": "freelance food writer",
    "sam": "rather be outside",
    # startup_csuite
    "priya": "second-time founder",
    "marcus": "ex-Boston Dynamics",
    "lila": "ex-Stripe ops lead",
    "devin": "ex-Bain",
    # van_life_detour
    "maddy": "laminated the itinerary",
    "theo": "photographer",
    "reyes": "labeled them",
    # heist_crew
    "nora": "sixteen years",
    "dane": "two hundred alarms",
    "rae": "youngest in the room",
    "jules": "eleven years",
    # family_thanksgiving
    "linda": "starting at 4am",
    "bob": "mostly retired",
    "sasha": "364 days ago",
    "jordan": "easy one",
    "mara": "comparative literature",
}


_SCRIPTS = {
    "trip_planning": TRIP_PLANNING_SCRIPT,
    "startup_csuite": STARTUP_CSUITE_SCRIPT,
    "van_life_detour": VAN_LIFE_DETOUR_SCRIPT,
    "heist_crew": HEIST_CREW_SCRIPT,
    "family_thanksgiving": FAMILY_THANKSGIVING_SCRIPT,
}


def make_mock_chat(scenario_name: str = "trip_planning") -> Any:
    """Return a chat fn that emulates the given scenario.

    `scenario_name` selects which canned script to use.
    """
    script = _SCRIPTS.get(scenario_name, TRIP_PLANNING_SCRIPT)
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

        # Missing entries → PASS so the scene wraps cleanly when the script
        # runs out, instead of every agent saying "...".
        line = script.get((agent_id, tick), "PASS")
        return llm.ChatResult(
            text=line, usage=_usage(), cost_usd=0.0008, model="claude-sonnet-4-6"
        )

    return fn
