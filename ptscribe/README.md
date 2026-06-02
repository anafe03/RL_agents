# ptscribe

**Ambient SOAP-note generation for PT / OT / ST visits.**

A clinician dictates the visit; ptscribe turns the loose-language transcript
into a structured SOAP note — Subjective, Objective with measurable
findings (ROM degrees, MMT grades, pain scores), Assessment, Plan with
specific exercises. Every run goes through a full eval harness:
hallucination cross-check against the transcript, section-completeness
metrics, LLM-as-judge for narrative quality, cost/latency monitoring.

```
visit transcript ──▶  Pydantic-validated extraction  ──▶  SOAPNote
                            │
                            ▼
                  eval harness  ──▶  ✓ hallucination check
                                    ✓ section completeness
                                    ✓ LLM-as-judge narrative
                                    ✓ cost / latency / run log
```

## Why this exists

PT/OT/ST documentation is one of the biggest sources of clinician
burnout in rehab therapy. A scribe that gets the structure right and
shows its work on quality is the difference between an AI tool clinicians
*use* and an AI tool they *audit*. The eval harness is the whole point:
the model proposes, the harness verifies — every claim in the SOAP note
gets cross-checked against the source transcript before it ships.

## Quickstart

```bash
cd ptscribe
uv sync
uv run streamlit run src/ptscribe/ui/app.py            # the demo + dashboard
uv run ptscribe scribe data/transcripts/knee_post_op.txt
uv run ptscribe eval                                    # run the harness
uv run uvicorn ptscribe.api:app --reload                # the FastAPI endpoint
```

## Architecture

| Module | Responsibility |
|---|---|
| `models.py`        | Pydantic SOAPNote + sub-models (ROM, MMT, pain, exercise) |
| `llm.py`           | Multi-provider chat (Anthropic + OpenAI), uncached clients |
| `prompts.py`       | The system prompt for SOAP extraction |
| `scribe.py`        | `extract_soap(transcript, model)` — the core pipeline |
| `hallucination.py` | Cross-check every extracted claim against the transcript |
| `eval.py`          | Golden dataset, section completeness, LLM-as-judge |
| `monitoring.py`    | SQLite run log — every call's cost, latency, output, eval |
| `mock.py`          | Bundled canned responses for demo mode (no API key) |
| `api.py`           | FastAPI `POST /scribe` endpoint |

## Demo vs. live

- **Demo mode** — a canned scribe + canned eval result for the bundled
  transcripts. No API key, fully exercisable in the hosted app.
- **Live mode** — your Anthropic or OpenAI key drives a real LLM. Every
  run is logged to SQLite with cost, latency, hallucination findings, and
  eval scores; the monitoring dashboard pages through them.

## Status

v0.1 — alpha. **A portfolio project, not a clinical tool.** Built to
demonstrate the eval-harness + observability discipline a production AI
team should ship with — not validated against real PT clinical practice.
SOAP examples are synthetic and the thresholds are illustrative.
