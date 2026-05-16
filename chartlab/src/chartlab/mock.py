"""Demo mode — a heuristic stand-in for the LLM and synthetic price data.

Lets the whole app run with no API key and no network: a keyword parser
fills in for the LLM, and price series are generated deterministically.
All demo prices are synthetic — clearly labelled as such in the UI.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from random import Random
from typing import Any

from chartlab import llm
from chartlab.models import PricePoint, Series

# --- synthetic price data ---------------------------------------------------

# ticker -> (start price, daily drift, daily volatility)
_SYNTH: dict[str, tuple[float, float, float]] = {
    "AAPL": (170.0, 0.0005, 0.017),
    "MSFT": (330.0, 0.0006, 0.015),
    "NVDA": (450.0, 0.0013, 0.030),
    "GOOGL": (135.0, 0.0004, 0.019),
    "TSLA": (210.0, 0.0003, 0.034),
    "AMZN": (145.0, 0.0005, 0.021),
    "SPY": (440.0, 0.0004, 0.009),
    "BTC-USD": (43000.0, 0.0010, 0.038),
}

DEMO_TICKERS: list[str] = list(_SYNTH)

_PERIOD_DAYS: dict[str, int] = {
    "1mo": 22, "3mo": 66, "6mo": 132, "ytd": 140,
    "1y": 252, "2y": 504, "5y": 900, "max": 900,
}


def _full_walk(ticker: str) -> list[float]:
    """A deterministic geometric-random-walk price history for a ticker."""
    start, drift, vol = _SYNTH.get(ticker.upper(), (100.0, 0.0004, 0.02))
    rng = Random(sum(ord(c) for c in ticker.upper()))
    prices: list[float] = []
    price = start
    for _ in range(900):
        price *= 1 + drift + rng.gauss(0, vol)
        prices.append(round(max(price, 0.01), 4))
    return prices


def demo_series(ticker: str, period: str) -> Series:
    """A synthetic `Series` for a ticker over the given period."""
    days = _PERIOD_DAYS.get(period, 252)
    closes = _full_walk(ticker)[-days:]
    today = date.today()
    points = [
        PricePoint(date=(today - timedelta(days=len(closes) - 1 - i)).isoformat(), close=c)
        for i, c in enumerate(closes)
    ]
    return Series(ticker=ticker.upper(), points=points)


# --- heuristic stand-in for the LLM ----------------------------------------

_COMPANY_TICKERS: dict[str, str] = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "tesla": "TSLA", "tsla": "TSLA",
    "amazon": "AMZN", "amzn": "AMZN",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "s&p 500": "SPY", "s&p": "SPY", "sp500": "SPY", "spy": "SPY",
}


def _heuristic_spec(text: str) -> dict[str, Any]:
    """Keyword-match a request into a ChartSpec dict — the demo's 'LLM'."""
    low = text.lower()

    tickers: list[str] = []
    for key, symbol in _COMPANY_TICKERS.items():  # longer keys first via insertion order
        if key in low and symbol not in tickers:
            tickers.append(symbol)
    if not tickers:
        tickers = ["AAPL"]

    if "ytd" in low or "this year" in low:
        period = "ytd"
    elif "6 month" in low or "six month" in low or "half year" in low:
        period = "6mo"
    elif "3 month" in low or "three month" in low or "quarter" in low:
        period = "3mo"
    elif "5 year" in low or "five year" in low:
        period = "5y"
    elif "2 year" in low or "two year" in low:
        period = "2y"
    elif "all time" in low or "max" in low:
        period = "max"
    elif "month" in low:
        period = "1mo"
    else:
        period = "1y"

    if any(w in low for w in ("index", "rebase", "normaliz")):
        transform = "indexed"
    elif any(w in low for w in ("percent", "% ", "growth", "return", "performance")):
        transform = "pct_change"
    elif len(tickers) > 1:
        transform = "indexed"  # fair comparison across price scales
    else:
        transform = "raw"

    title = " vs ".join(tickers) if len(tickers) > 1 else tickers[0]
    return {"tickers": tickers, "period": period, "transform": transform, "title": title}


def make_mock_chat():
    """A chat function (for `llm.set_chat_fn`) that returns a heuristic spec."""

    def _mock(*, model: str, system: str, messages: list[dict], max_tokens: int = 300,
              **_: Any) -> llm.ChatResult:
        content = messages[-1].get("content", "") if messages else ""
        text = content if isinstance(content, str) else str(content)
        return llm.ChatResult(
            text=json.dumps(_heuristic_spec(text)),
            usage=None,
            cost_usd=0.0,
            model=model,
        )

    return _mock
