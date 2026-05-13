# anafe03 — portfolio monorepo

Five LLM-agent projects, each independently deployable, each demoable in mock mode without an API key. Built to demonstrate **RAG**, **multi-agent orchestration**, **agent safety / guardrails**, and **structured-extraction-with-citations** across distinct domains.

## Projects

| Project | One-line | Themes | Demo |
|---|---|---|---|
| [**Octagon**](./octagon) — *Red Cell* | Adversarial audit for LLM agents. Library of attacks (prompt injection, social engineering, indirect injection) pitted against a defender; LLM-as-judge verdict; pen-test-style report. | AI agents, cyberrisk | mock + live |
| [**Simulacrum**](./simulacrum) | Multi-agent scenario engine. Cast personas, set a stage, watch the agents argue / plan / improvise. Five scenarios: trip planning, c-suite Series A, vanlife detour, heist crew, family Thanksgiving. | AI agents, simulation | mock + live |
| [**Festival Companion**](./festival) | Personalized festival schedule planner. Taste-matched set picks with conflict resolution via weighted-interval scheduling (real CS, not LLM-wrapped). | creative/music, adventure/travel | mock + live |
| [**PriorAuth Assist**](./priorauth) | Cited insurance prior-auth appeal drafter with three pluggable retrieval backends (BM25, ChromaDB, LLM-judged) and a built-in benchmark harness. Citation enforcement guarantees every clinical claim traces to a real guideline. | healthcare, AI agents | mock + live |
| [**Earnings Call Inspector**](./earningscall) | Multi-pass extraction over earnings transcripts — metrics, tone, surprises, analyst-question sharpness — every claim cited back to the transcript with substring verification. | finance, AI agents | mock + live |

## Deploying

Each project is independently deployable to Streamlit Community Cloud. From any project's directory the local UI runs with:

```bash
cd <project>
uv sync
uv run streamlit run src/<project>/ui/app.py
```

For Streamlit Cloud deployment, point the app at the subproject's `src/<project>/ui/app.py`. The root [`requirements.txt`](./requirements.txt) installs all five packages so any of them can deploy from the monorepo root.

## Repo layout

```
RL_agents/
├── octagon/           — adversarial audit + tournament platform
├── simulacrum/        — multi-agent scenario engine
├── festival/          — festival schedule planner
├── priorauth/         — cited prior-auth appeal drafter (with RAG benchmark)
├── earningscall/      — earnings transcript inspector
├── requirements.txt   — installs all five for Streamlit Cloud
├── .streamlit/        — shared theme
└── README.md          — this file
```

Each subproject has its own `pyproject.toml`, `README.md`, tests, and synthetic data. The monorepo structure exists to share deploy plumbing; each project can be lifted into its own GitHub repo with `git subtree split` whenever portfolio links call for it.
