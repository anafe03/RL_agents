"""Tests for chartlab — models, NL parsing, transforms, demo data, charting.

The real LLM and live Yahoo Finance fetch are not tested (they need a key
and network). Everything else — the spec coercion, the demo heuristic, the
transforms, the figure build — is pure and fully exercised here.
"""

from __future__ import annotations

from chartlab import llm
from chartlab.chart import apply_transform, build_figure, summarize
from chartlab.mock import DEMO_TICKERS, _heuristic_spec, demo_series, make_mock_chat
from chartlab.models import ChartSpec, PricePoint, Series, Transform
from chartlab.nlp import _coerce_spec, parse_request


def _series(ticker: str, closes: list[float]) -> Series:
    return Series(
        ticker=ticker,
        points=[PricePoint(date=f"2025-01-{i + 1:02d}", close=c) for i, c in enumerate(closes)],
    )


# -- models ------------------------------------------------------------------

def test_chartspec_normalized_uppercases_and_clamps():
    spec = ChartSpec(tickers=["aapl", " msft "], period="banana")
    norm = spec.normalized()
    assert norm.tickers == ["AAPL", "MSFT"]
    assert norm.period == "1y"  # invalid period clamped


def test_chartspec_keeps_valid_period():
    assert ChartSpec(tickers=["AAPL"], period="6mo").normalized().period == "6mo"


# -- nlp spec coercion -------------------------------------------------------

def test_coerce_spec_parses_clean_json():
    spec = _coerce_spec('{"tickers": ["AAPL"], "period": "1y", "transform": "indexed"}')
    assert spec.tickers == ["AAPL"]
    assert spec.transform == Transform.INDEXED


def test_coerce_spec_extracts_json_from_prose():
    spec = _coerce_spec('Sure! {"tickers": ["msft"], "period": "ytd", "transform": "raw"} done')
    assert spec.tickers == ["MSFT"]
    assert spec.period == "ytd"


def test_coerce_spec_falls_back_on_bad_values():
    spec = _coerce_spec('{"tickers": ["x"], "period": "nonsense", "transform": "weird"}')
    assert spec.period == "1y"
    assert spec.transform == Transform.RAW


# -- demo heuristic ----------------------------------------------------------

def test_heuristic_detects_tickers_and_comparison():
    spec = _heuristic_spec("compare apple and microsoft this year")
    assert spec["tickers"] == ["AAPL", "MSFT"]
    assert spec["period"] == "ytd"
    assert spec["transform"] == "indexed"  # multi-ticker -> indexed


def test_heuristic_detects_percent():
    spec = _heuristic_spec("how has nvidia done in percent")
    assert spec["tickers"] == ["NVDA"]
    assert spec["transform"] == "pct_change"


def test_heuristic_defaults_when_nothing_matches():
    spec = _heuristic_spec("show me a chart")
    assert spec["tickers"] == ["AAPL"]
    assert spec["period"] == "1y"


# -- demo series -------------------------------------------------------------

def test_demo_series_length_matches_period():
    assert len(demo_series("AAPL", "1mo").points) == 22
    assert len(demo_series("MSFT", "1y").points) == 252


def test_demo_series_is_deterministic():
    a = demo_series("NVDA", "3mo").closes
    b = demo_series("NVDA", "3mo").closes
    assert a == b


def test_demo_tickers_all_generate():
    for ticker in DEMO_TICKERS:
        assert len(demo_series(ticker, "6mo").points) == 132


# -- transforms --------------------------------------------------------------

def test_apply_transform_raw():
    s = _series("AAPL", [100.0, 110.0, 90.0])
    assert apply_transform(s, Transform.RAW) == [100.0, 110.0, 90.0]


def test_apply_transform_indexed():
    s = _series("AAPL", [50.0, 75.0, 100.0])
    assert apply_transform(s, Transform.INDEXED) == [100.0, 150.0, 200.0]


def test_apply_transform_pct_change():
    s = _series("AAPL", [100.0, 110.0, 80.0])
    assert apply_transform(s, Transform.PCT_CHANGE) == [0.0, 10.0, -20.0]


def test_summarize():
    s = _series("AAPL", [100.0, 120.0])
    out = summarize(s, Transform.RAW)
    assert out["start"] == 100.0
    assert out["end"] == 120.0
    assert out["change_pct"] == 20.0


# -- charting ----------------------------------------------------------------

def test_build_figure_has_one_trace_per_series():
    spec = ChartSpec(tickers=["AAPL", "MSFT"], period="1y", transform=Transform.INDEXED)
    fig = build_figure(spec, [_series("AAPL", [1.0, 2.0]), _series("MSFT", [3.0, 4.0])])
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"AAPL", "MSFT"}


# -- end to end with the mock LLM -------------------------------------------

def test_parse_request_with_mock_chat():
    llm.set_chat_fn(make_mock_chat())
    try:
        spec, cost = parse_request("bitcoin vs the s&p 500 indexed to 100")
    finally:
        llm.reset_chat_fn()
    assert "BTC-USD" in spec.tickers
    assert "SPY" in spec.tickers
    assert spec.transform == Transform.INDEXED
    assert cost == 0.0  # mock is free


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
