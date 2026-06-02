"""Run monitoring — every scribe call gets logged with cost, latency, and eval result.

Uses SQLAlchemy 2.0 against a local SQLite file (zero setup) by default.
Pointing it at Postgres for production is a one-line URL change — same
ORM, same code path.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ptscribe.models import EvalResult, RunRecord

_DEFAULT_DB = Path.home() / ".ptscribe" / "runs.sqlite"


def _resolve_url(url: str | None = None) -> str:
    if url:
        return url
    env = os.environ.get("PTSCRIBE_DB_URL")
    if env:
        return env
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_DB}"


class _Base(DeclarativeBase):
    pass


class _RunRow(_Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column()
    transcript_id: Mapped[str] = mapped_column(String, default="")
    model: Mapped[str] = mapped_column(String, default="")
    mode: Mapped[str] = mapped_column(String, default="demo")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_chars: Mapped[int] = mapped_column(Integer, default=0)
    output_chars: Mapped[int] = mapped_column(Integer, default=0)
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0)
    hallucination_count: Mapped[int] = mapped_column(Integer, default=0)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(String, default="")


_engine = None


def get_engine(url: str | None = None):
    """Lazily build (and memoize) the SQLAlchemy engine + schema."""
    global _engine
    if _engine is None:
        _engine = create_engine(_resolve_url(url), echo=False, future=True)
        _Base.metadata.create_all(_engine)
    return _engine


def reset_for_test(url: str = "sqlite:///:memory:") -> None:
    """Re-point the engine at a fresh in-memory DB. For tests only."""
    global _engine
    _engine = create_engine(url, echo=False, future=True)
    _Base.metadata.create_all(_engine)


def log_run(record: RunRecord) -> int:
    """Persist one RunRecord. Returns the new row id."""
    engine = get_engine()
    with Session(engine) as session:
        row = _RunRow(
            timestamp=record.timestamp,
            transcript_id=record.transcript_id,
            model=record.model,
            mode=record.mode,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
            input_chars=record.input_chars,
            output_chars=record.output_chars,
            completeness_score=record.completeness_score,
            hallucination_count=record.hallucination_count,
            judge_score=record.judge_score,
            error=record.error,
        )
        session.add(row)
        session.commit()
        return int(row.id)


def log_from_eval(
    *,
    transcript_id: str,
    model: str,
    mode: str,
    cost_usd: float,
    latency_ms: int,
    input_chars: int,
    output_chars: int,
    eval_result: EvalResult,
    error: str = "",
) -> int:
    """Convenience constructor: assemble a RunRecord from an EvalResult and log it."""
    return log_run(RunRecord(
        transcript_id=transcript_id,
        model=model,
        mode=mode,
        cost_usd=round(cost_usd, 6),
        latency_ms=latency_ms,
        input_chars=input_chars,
        output_chars=output_chars,
        completeness_score=eval_result.completeness_score,
        hallucination_count=len(eval_result.hallucination_findings),
        judge_score=eval_result.judge_score,
        error=error,
    ))


def recent_runs(limit: int = 50) -> list[RunRecord]:
    """Most-recent-first run history for the dashboard."""
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(_RunRow).order_by(_RunRow.timestamp.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [
            RunRecord(
                id=str(r.id),
                timestamp=r.timestamp,
                transcript_id=r.transcript_id,
                model=r.model,
                mode=r.mode,
                cost_usd=r.cost_usd,
                latency_ms=r.latency_ms,
                input_chars=r.input_chars,
                output_chars=r.output_chars,
                completeness_score=r.completeness_score,
                hallucination_count=r.hallucination_count,
                judge_score=r.judge_score,
                error=r.error,
            )
            for r in rows
        ]


def aggregate_stats() -> dict:
    """Roll-up stats for the dashboard: counts, p50/p95 latency, avg cost, etc."""
    engine = get_engine()
    with Session(engine) as session:
        total = session.execute(select(func.count(_RunRow.id))).scalar_one()
        if total == 0:
            return {"total_runs": 0}
        avg_cost = session.execute(select(func.avg(_RunRow.cost_usd))).scalar_one() or 0.0
        avg_latency = session.execute(select(func.avg(_RunRow.latency_ms))).scalar_one() or 0
        avg_completeness = session.execute(
            select(func.avg(_RunRow.completeness_score))
        ).scalar_one() or 0.0
        total_hallucinations = session.execute(
            select(func.sum(_RunRow.hallucination_count))
        ).scalar_one() or 0
        # SQLite has no percentile_cont; pull latencies and compute in Python.
        latencies = session.execute(
            select(_RunRow.latency_ms).order_by(_RunRow.latency_ms.asc())
        ).scalars().all()
        p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0
        return {
            "total_runs": int(total),
            "avg_cost_usd": round(float(avg_cost), 5),
            "avg_latency_ms": int(avg_latency),
            "p50_latency_ms": int(p50),
            "p95_latency_ms": int(p95),
            "avg_completeness": round(float(avg_completeness), 3),
            "hallucinations_total": int(total_hallucinations),
            "hallucinations_per_run": round(
                float(total_hallucinations) / float(total), 3
            ),
        }
