"""Core data models for chartlab.

A `ChartSpec` is what the LLM produces from a natural-language request —
the precise, validated parameters the charting code needs. `Series` is the
price data fetched for one ticker.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Periods accepted by yfinance's `history(period=...)`.
PERIODS: list[str] = ["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "max"]


class Transform(str, Enum):
    """How series values are presented."""

    RAW = "raw"  # actual prices
    INDEXED = "indexed"  # each series rebased to 100 at its start
    PCT_CHANGE = "pct_change"  # cumulative % change from the start


class PricePoint(BaseModel):
    date: str  # ISO date, e.g. "2025-03-14"
    close: float


class Series(BaseModel):
    """Price history for a single ticker."""

    ticker: str
    points: list[PricePoint] = Field(default_factory=list)

    @property
    def dates(self) -> list[str]:
        return [p.date for p in self.points]

    @property
    def closes(self) -> list[float]:
        return [p.close for p in self.points]


class ChartSpec(BaseModel):
    """The structured chart request — the LLM's output, validated."""

    tickers: list[str] = Field(default_factory=list)
    period: str = "1y"
    transform: Transform = Transform.RAW
    title: str = ""

    def normalized(self) -> ChartSpec:
        """Return a copy with tickers upper-cased and the period clamped."""
        tickers = [t.strip().upper() for t in self.tickers if t.strip()]
        period = self.period if self.period in PERIODS else "1y"
        return self.model_copy(update={"tickers": tickers, "period": period})
