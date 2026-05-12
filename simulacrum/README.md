# Simulacrum

> Cast a few personas, set a stage, press play.

**🎭 [Try the live demo →](https://simulacrum-scenes.streamlit.app/)** *(deploy URL — replace once published)*

**Simulacrum** is a multi-agent scenario engine. You define a small cast of agents (each with a persona, goals, and memories), set an initial situation, and the engine runs N "ticks" of simulated time during which the agents interact, react, and converge — or fail to — on whatever the scenario asks of them.

Use it to watch a planning meeting unfold, a road trip get debated, a startup leadership team fight about Series A focus, or an ER team triage during a busy night.

The hosted demo runs in mock mode by default — pick a scenario, press ▶ Play, and watch a canned-but-realistic dialogue unfold in 2 seconds, no API key required. Switch to "Live" mode in the sidebar to run real LLMs (Claude Sonnet 4.6) with your own `ANTHROPIC_API_KEY`.

## Five scenarios ship out of the box

- **`trip_planning`** — Three friends (a budget-watcher, a foodie, an adventure-seeker) plan a 4-day Lisbon trip. Watch how the group negotiates between cost, food, and itinerary density.
- **`startup_csuite`** — A four-person founding team (CEO, CTO, COO, CFO) walks into a Monday morning Series A planning meeting with conflicting priorities.
- **`van_life_detour`** — Three friends in a Sprinter van get their planned route to Mt. Rainier closed by a wildfire on day 4. Twenty minutes to pick a new destination. Trip planner, photographer, and gear-and-food logistics specialist do not agree on what matters.
- **`heist_crew`** — Friday 11pm, the night before the job. Mastermind, safecracker, getaway driver, and the inside contact run the final pre-execution review. Last chance for cold feet. *(Cinematic.)*
- **`family_thanksgiving`** — Mom cooked everything, Dad is mostly retired, both kids are home, and one of them brought their new partner — meeting the family for the first time, the year after a complicated coming-out. Five voices around a dry turkey. *(Relatable.)*

## Why this exists

Static agent benchmarks tell you whether one agent can answer one question. They don't tell you how agents behave *over time*, *in relation to each other*, *in the presence of conflicting goals*. Simulacrum is the cheap, fast version of that — no production deployment, no real users, no real money. Just a roster, a scenario, a transcript.

## Architecture

```
simulacrum/
├── src/simulacrum/
│   ├── models.py        ← Agent, Persona, Scenario, Event, Tick, Transcript
│   ├── llm.py           ← anthropic SDK wrapper with prompt caching
│   ├── engine.py        ← the tick loop: observe → decide → act → record
│   ├── memory.py        ← per-agent memory store (short-term + reflections)
│   ├── render.py        ← terminal + markdown transcript
│   └── cli.py
├── scenarios/
│   ├── trip_planning/
│   │   ├── scenario.yaml
│   │   └── personas/
│   │       ├── alex.md     ← budget-watcher
│   │       ├── maya.md     ← foodie
│   │       └── sam.md      ← adventure-seeker
│   └── startup_csuite/
│       └── ...
└── tests/
```

A **scenario** is a folder. To add a new one, drop in a `scenario.yaml` and a `personas/` directory. No code changes.

## Quick start — local UI

```bash
cd simulacrum
uv sync
uv run streamlit run src/simulacrum/ui/app.py
# → opens at http://localhost:8501; mock mode works without an API key
```

## Quick start — CLI

```bash
cd simulacrum
uv sync
export ANTHROPIC_API_KEY=...

simulacrum list-scenarios
simulacrum run scenarios/trip_planning --ticks 12 --transcript trip.md
```

## Deploy your own

The `simulacrum/` directory ships ready for [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your account, and pick this repo.
3. Set **main file path** to `simulacrum/src/simulacrum/ui/app.py` and Python version to `3.11`.
4. Set **app directory** (or "Working directory" in Advanced settings) to `simulacrum/` so the `requirements.txt` and `scenarios/` are found.
5. *(Optional)* Add `ANTHROPIC_API_KEY` in *Advanced settings → Secrets* to allow visitors to use Live mode without entering their own key.
6. Click Deploy.

## Roadmap

- **v0.1** *(current)* — Scenario loader + tick engine + 2 ship-with scenarios + markdown transcripts.
- **v0.2** — Per-agent reflective memory ("at end of day, summarize what mattered").
- **v0.3** — Event injection ("a storm rolls in," "an investor offers term sheet").
- **v0.4** — Streamlit visualization: agent roster, dialogue feed, day scrubber.

## License

MIT.
