"""Price data — fetched live from Yahoo Finance via yfinance.

Live mode only. Demo mode uses the synthetic series in `mock.py`. yfinance
needs network and can be rate-limited, so callers should handle failure.
"""

from __future__ import annotations

from chartlab.models import PricePoint, Series


def fetch_series(ticker: str, period: str) -> Series:
    """Fetch close-price history for one ticker. Raises on failure."""
    import yfinance

    hist = yfinance.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker!r} (unknown symbol?).")
    points = [
        PricePoint(date=str(idx.date()), close=round(float(row["Close"]), 4))
        for idx, row in hist.iterrows()
    ]
    return Series(ticker=ticker.upper(), points=points)


def fetch_all(tickers: list[str], period: str) -> tuple[list[Series], list[str]]:
    """Fetch every ticker. Returns (series, failed_tickers) — one bad symbol
    does not sink the rest."""
    series: list[Series] = []
    failed: list[str] = []
    for ticker in tickers:
        try:
            series.append(fetch_series(ticker, period))
        except Exception:  # noqa: BLE001 - network/symbol errors are per-ticker
            failed.append(ticker)
    return series, failed
