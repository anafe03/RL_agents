# anafe03 — portfolio monorepo

Seven LLM-agent projects, each independently deployable, each demoable in mock mode without an API key. Built to demonstrate **RAG**, **multi-agent orchestration**, **agent safety / guardrails**, **structured extraction with citation enforcement**, **Computer Use / browser automation**, and **post-training with verifiable rewards (GRPO)** across distinct domains.

## Projects

| Project | One-line | Themes | Demo |
|---|---|---|---|
| [**Octagon**](./octagon) — *Red Cell* | Adversarial audit for LLM agents. Library of attacks (prompt injection, social engineering, indirect injection) pitted against a defender; LLM-as-judge verdict; pen-test-style report. | AI agents, cyberrisk | mock + live |
| [**Simulacrum**](./simulacrum) | Multi-agent scenario engine. Five scenarios across travel, business, drama, family. | AI agents, simulation | mock + live |
| [**Festival Companion**](./festival) | Personalized festival schedule planner. Taste-matched picks with conflict resolution via weighted-interval scheduling. | creative/music, adventure | mock + live |
| [**PriorAuth Assist**](./priorauth) | Cited insurance prior-auth appeal drafter with three pluggable retrieval backends + benchmark + PHI de-identification. | healthcare, AI agents | mock + live |
| [**Earnings Call Inspector**](./earningscall) | Multi-pass extraction over earnings transcripts — metrics, tone, surprises, analyst Q&A sharpness — every claim cited with substring verification. | finance, AI agents | mock + live |
| [**AutoFill**](./autofill) | The loop after PriorAuth: Computer Use agent that submits the structured appeal to a real public-record state insurance commissioner complaint form. Anthropic Computer Use + Playwright. | healthcare, AI agents, automation | mock playback |
| [**tunelab**](./tunelab) | Post-training experiments. GRPO fine-tunes a small open model with verifiable rewards on a structured-extraction task. | RL, post-training | reward + eval pieces; training is GPU-only |

## What's in this repo, technically

Across these projects you can find concrete implementations of:

- **RAG** with three swappable backends benchmarked against per-case golden sets ([`priorauth/retrievers/`](priorauth/src/priorauth/retrievers))
- **Citation enforcement** — every clinical claim or financial metric is verified as a real substring of the source ([`priorauth`](priorauth/src/priorauth/drafter.py), [`earningscall/verifier.py`](earningscall/src/earningscall/verifier.py))
- **Multi-pass structured extraction** — separate LLM calls per analytic angle, each with its own focused system prompt + Pydantic-typed output ([`earningscall/extractor.py`](earningscall/src/earningscall/extractor.py))
- **Drafter / assessor split** — two LLM runs with different objectives (be persuasive vs. find weaknesses) instead of one ([`priorauth`](priorauth/src/priorauth/assessor.py))
- **Weighted-interval scheduling** — classic CS algorithm doing real work in a non-LLM-wrapper part of the pipeline ([`festival/scheduler.py`](festival/src/festival/scheduler.py))
- **Anthropic Computer Use** driving a real browser via Playwright with hard "halt before submit" rule ([`autofill/agent.py`](autofill/src/autofill/agent.py))
- **PHI de-identification** with structured + regex-driven scrubbing and reverse-mapping for re-identification ([`priorauth/deidentify.py`](priorauth/src/priorauth/deidentify.py))
- **Verifiable-reward post-training** (GRPO) — reward function design, eval harness, training script ready for Colab/GPU ([`tunelab`](tunelab))
- **Optional LangSmith observability** as a no-op decorator that activates when the env vars are set ([`priorauth/observability.py`](priorauth/src/priorauth/observability.py))
- **Mock-mode everywhere** — every project ships a deterministic mock that runs the full pipeline shape without an API key, so the public demo works zero-config

## Deploying

Each project is independently deployable to Streamlit Community Cloud. From any project's directory the local UI runs with:

```bash
cd <project>
uv sync
uv run streamlit run src/<project>/ui/app.py
```

For Streamlit Cloud deployment, point the app at the subproject's `src/<project>/ui/app.py`. The root [`requirements.txt`](./requirements.txt) installs all seven packages so any of them can deploy from the monorepo root.

## Repo layout

```
RL_agents/
├── octagon/           — adversarial audit + tournament platform
├── simulacrum/        — multi-agent scenario engine
├── festival/          — festival schedule planner
├── priorauth/         — cited prior-auth appeal drafter (RAG benchmark, de-id)
├── earningscall/      — earnings transcript inspector
├── autofill/          — Computer Use agent for public complaint forms
├── tunelab/           — post-training experiments (GRPO + verifiable rewards)
├── requirements.txt   — installs all seven for Streamlit Cloud
├── .streamlit/        — shared theme
└── README.md          — this file
```

Each subproject has its own `pyproject.toml`, `README.md`, tests, and synthetic data. The monorepo structure exists to share deploy plumbing; each project can be lifted into its own GitHub repo with `git subtree split` whenever portfolio links call for it.
