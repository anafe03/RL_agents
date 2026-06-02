"""FastAPI endpoint — `POST /scribe` for production-style use.

A thin wrapper over the same `extract_soap` + `run_eval` + `log_run`
pipeline the CLI and UI use. The request body carries the transcript and
options; the response is the structured SOAP note plus the eval result.

Run with:
    uv run uvicorn ptscribe.api:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ptscribe.eval import run_eval
from ptscribe.models import EvalResult, SOAPNote
from ptscribe.monitoring import aggregate_stats, log_from_eval, recent_runs
from ptscribe.scribe import extract_soap


class ScribeRequest(BaseModel):
    transcript: str = Field(..., min_length=20)
    model: str = "claude-sonnet-4-6"
    use_judge: bool = False
    transcript_id: str = ""


class ScribeResponse(BaseModel):
    note: SOAPNote
    eval: EvalResult
    cost_usd: float
    latency_ms: int


app = FastAPI(
    title="ptscribe",
    summary="Ambient SOAP-note generation for PT/OT/ST visits.",
    version="0.0.1",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scribe", response_model=ScribeResponse)
def scribe(req: ScribeRequest) -> ScribeResponse:
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="Set ANTHROPIC_API_KEY or OPENAI_API_KEY before calling /scribe.",
        )
    try:
        note, chat_result = extract_soap(req.transcript, model=req.model)
        eval_result, judge_cost = run_eval(
            req.transcript, note,
            transcript_id=req.transcript_id,
            use_judge=req.use_judge, judge_model=req.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    total_cost = chat_result.cost_usd + judge_cost
    log_from_eval(
        transcript_id=req.transcript_id,
        model=chat_result.model,
        mode="live",
        cost_usd=total_cost,
        latency_ms=chat_result.latency_ms,
        input_chars=len(req.transcript),
        output_chars=len(chat_result.text),
        eval_result=eval_result,
    )
    return ScribeResponse(
        note=note,
        eval=eval_result,
        cost_usd=round(total_cost, 6),
        latency_ms=chat_result.latency_ms,
    )


@app.get("/runs")
def runs(limit: int = 50) -> dict:
    return {"runs": [r.model_dump(mode="json") for r in recent_runs(limit=limit)]}


@app.get("/stats")
def stats() -> dict:
    return aggregate_stats()
