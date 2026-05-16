"""Natural-language request → ChartSpec.

The LLM step. It maps a loose English request onto the precise parameters
the charting code needs — which tickers, what time window, and how to
present the series (raw prices, indexed to 100, or % change).
"""

from __future__ import annotations

import json

from chartlab import llm
from chartlab.models import PERIODS, ChartSpec, Transform

SYSTEM_PROMPT = """You convert a plain-English chart request into a JSON chart spec.

Respond with ONLY a JSON object, no prose:
{"tickers": ["AAPL", "MSFT"], "period": "1y", "transform": "indexed", "title": "..."}

Rules:
- tickers: uppercase market symbols. Map company names to symbols
  (Apple->AAPL, Microsoft->MSFT, Nvidia->NVDA, Google/Alphabet->GOOGL,
  Tesla->TSLA, Amazon->AMZN, Bitcoin->BTC-USD, "the S&P"/"S&P 500"->SPY).
- period: one of 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, max. Map "this year"->ytd,
  "last year"->1y, "past 6 months"->6mo. Default 1y if unspecified.
- transform: one of raw, indexed, pct_change.
  - "indexed" when comparing multiple tickers (fair comparison across
    different price scales) or when the user says "indexed"/"rebased".
  - "pct_change" when the user asks about growth, returns, performance,
    or "in percent".
  - "raw" otherwise.
- title: a short, descriptive chart title.
"""


def _coerce_spec(raw_text: str) -> ChartSpec:
    """Parse the LLM's JSON reply into a validated ChartSpec."""
    text = raw_text.strip()
    obj: dict
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON: {raw_text[:200]!r}") from None
        obj = json.loads(text[start : end + 1])

    transform_raw = str(obj.get("transform", "raw")).lower()
    transform = (
        Transform(transform_raw)
        if transform_raw in {t.value for t in Transform}
        else Transform.RAW
    )
    period = str(obj.get("period", "1y"))
    spec = ChartSpec(
        tickers=[str(t) for t in obj.get("tickers", [])],
        period=period if period in PERIODS else "1y",
        transform=transform,
        title=str(obj.get("title", "")),
    )
    return spec.normalized()


def parse_request(text: str, model: str = "claude-sonnet-4-6") -> tuple[ChartSpec, float]:
    """Turn a natural-language request into a ChartSpec. Returns (spec, cost_usd)."""
    result = llm.chat(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        max_tokens=300,
    )
    return _coerce_spec(result.text), result.cost_usd
