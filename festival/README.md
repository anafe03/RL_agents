# Festival Companion

> Paste a lineup, paste your taste, get a schedule.

**Festival Companion** plans your festival weekend. Drop in a lineup (artists × stages × time slots), describe what you like, and the agent ranks every set by taste-fit, then runs a **weighted interval scheduler** to pick the best non-overlapping picks per day — calling out the great sets you had to skip due to conflicts.

**🎟️ [Try the live demo →](https://festival-companion.streamlit.app/)** *(deploy URL — replace once published)*

## The engineering wedge

This is not a "list your favorite artists, GPT picks them" demo. The interesting work is the scheduler.

For each day, you have N sets, each with a `(stage, start, end)` and a taste-match score. Many overlap in time across stages. The picker has to choose a non-overlapping subset that maximizes total score *while* respecting hard constraints like must-sees.

This is **weighted interval scheduling** — a classic CS problem solvable in O(n log n) with DP after sorting by end time. We implement it from scratch in [`scheduler.py`](src/festival/scheduler.py), then layer the LLM on top to write the human-readable "why this fits you" annotations for each pick.

The split is deliberate:
- **Deterministic:** scoring → scheduling → output structure.
- **LLM:** turning a score into prose, turning a profile blurb into per-artist signal, turning "skipped due to conflict" into useful regret notes.

## Quick start — local UI

```bash
cd festival
uv sync
uv run streamlit run src/festival/ui/app.py
# → opens at http://localhost:8501; mock mode works without an API key
```

## Quick start — CLI

```bash
cd festival
uv sync
export ANTHROPIC_API_KEY=...

festival list-lineups
festival plan data/lineups/skyline_2025.yaml --taste "indie rock, dream pop, electronic; favorites: Boygenius, Beach House, Caribou" --report my-plan.md
```

## Deploy your own

Same Streamlit Cloud flow as the rest of the repo — see the top-level [README](../README.md#deploy) for the click-through. Main file path: `festival/src/festival/ui/app.py`.

## How a lineup looks

```yaml
# festival/data/lineups/skyline_2025.yaml
name: Skyline 2025
city: San Francisco, CA
year: 2025
days: [Friday, Saturday, Sunday]
stages: [Lands End, Polo Field, Sutro, Twin Peaks]

sets:
  - artist: Phoebe Bridgers
    stage: Polo Field
    day: Friday
    start: "20:30"
    end: "21:45"
    headliner: true
  - artist: Caribou
    stage: Twin Peaks
    day: Friday
    start: "20:15"
    end: "21:30"
  # ...
```

Drop in a new YAML, the CLI and UI pick it up.

## Roadmap

- **v0.1** *(current)* — Hand-curated lineups + paste-your-taste profile + weighted-interval scheduler + LLM annotations + Streamlit UI.
- **v0.2** — Spotify OAuth for automatic taste extraction from listening history + audio features.
- **v0.3** — Walk-time estimates between stages when a venue map / coordinates are provided.
- **v0.4** — Group mode: blend tastes across multiple users, find the shared schedule.

## License

MIT.
