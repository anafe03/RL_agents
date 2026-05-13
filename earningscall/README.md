# Earnings Call Inspector

> Paste a transcript. Get an analyst-grade breakdown, every claim cited.

**📊 [Try the live demo →](https://earnings-inspector.streamlit.app/)** *(deploy URL — replace once published)*

Public-company earnings calls run 45-60 minutes and produce 5,000-word transcripts. Sell-side analysts read them by 8am for a reason — that's where the real signal is. **Earnings Call Inspector** runs an LLM pipeline that reads a transcript and produces four structured artifacts an analyst would build in their notebook:

1. **Key metrics** — revenue, EPS, guidance, segment results, surfaced with the line they came from.
2. **Tone assessment** — per speaker, per segment. Cautious / defensive / optimistic / evasive, with evidence quotes.
3. **Surprises** — guidance revisions, new initiatives, leadership changes, big-picture pivots that weren't in the prepared remarks.
4. **Analyst Q&A ranking** — ranked by sharpness (which analyst actually pushed for a substantive answer? whose question got dodged?).

Every insight has a citation back to a specific line in the transcript. **No claim without a quote.** A test verifies that every cited quote is a real substring of the input transcript — if the model hallucinates a quote, the pipeline flags it before the report is served.

## The engineering wedge

Multi-pass structured extraction with citation enforcement. Same pattern as PriorAuth Assist but for earnings calls:

- **One LLM call per analytic pass** rather than asking for everything at once. Each pass has its own focused system prompt, structured-output schema, and verification step.
- **Citation enforcement.** Each `Metric`, `ToneAssessment`, `Surprise`, and `AnalystQuestion` carries a quoted excerpt. Before the report is finalized, every quote is verified to be a substring of the input transcript. Hallucinated quotes are rejected.
- **Speaker-aware parsing.** The transcript is segmented into structured turns (CEO / CFO / Operator / Analyst) before extraction, so passes that operate per-speaker (tone, Q&A) get clean inputs.

## Two synthetic transcripts ship

Both fictional companies, both crafted to demonstrate interesting outputs:

- **`glow_q4_2025`** — "Glow Therapeutics" misses revenue, cuts FY guidance, CFO is defensive, analysts push hard. Demonstrates surprises + ranked Q&A.
- **`helios_q3_2025`** — "Helios Robotics" mixed quarter — beats revenue, lowers margin guidance, CEO is upbeat, CFO is measured. Demonstrates contrasting per-speaker tone.

## Quick start — local UI

```bash
cd earningscall
uv sync
uv run streamlit run src/earningscall/ui/app.py
# Demo mode works without an API key.
```

## Quick start — CLI

```bash
cd earningscall
uv sync
export ANTHROPIC_API_KEY=...

earnings list-transcripts
earnings inspect data/transcripts/glow_q4_2025.yaml --report glow.md
```

## Architecture

```
earningscall/
├── src/earningscall/
│   ├── models.py        ← Pydantic: Transcript, SpeakerTurn, Quote, Metric,
│   │                                ToneAssessment, Surprise, AnalystQuestion,
│   │                                EarningsReport
│   ├── llm.py           ← Anthropic SDK wrapper with prompt caching
│   ├── parser.py        ← raw transcript → structured speaker turns
│   ├── extractor.py     ← 4-pass extraction (metrics / tone / surprises / Q&A)
│   ├── verifier.py      ← citation-substring check before serving the report
│   ├── mock.py          ← demo-mode chat function (no API key)
│   ├── cli.py
│   └── ui/app.py
└── data/transcripts/    ← synthetic earnings transcript YAMLs
```

## Roadmap

- **v0.1** *(current)* — 2 transcripts, 4-pass extractor, citation verification, Streamlit UI with per-pass tabs.
- **v0.2** — Multi-quarter mode: compare this quarter against the prior 4 calls, surface what's NEW in tone/themes.
- **v0.3** — Live transcript ingestion from a URL (e.g., Seeking Alpha / Motley Fool format).
- **v0.4** — Watchlist mode: track a portfolio of tickers, get a daily summary of which calls happened and what mattered.

## Disclaimers

This is a portfolio project. **It is not investment advice, not a substitute for primary research, and not authorized for use in trading decisions.** All transcripts shipped here are synthetic for demonstration. Real use against real transcripts requires the user to validate every cited quote.

## License

MIT.
