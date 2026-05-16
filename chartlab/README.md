# chartlab

**Natural-language financial charting agent.**

Describe the chart you want in plain English — *"compare AAPL and MSFT over
the last year, indexed to 100"* — and chartlab uses an LLM to turn that
into a structured chart spec, fetches the price data, and plots it.

```
"how has Nvidia done vs the S&P this year, in percent"
        │
        ▼  LLM parses → ChartSpec
   { tickers: [NVDA, SPY], period: ytd, transform: pct_change }
        │
        ▼  fetch prices → transform → plot
   an interactive comparison chart
```

## Why an LLM here

The interesting problem is the gap between how people *describe* a chart
and the structured parameters a plotting library needs. "Indexed to 100",
"in percent", "this year", "vs the S&P" — chartlab's LLM step maps loose
natural language onto a precise `ChartSpec` (tickers, period, transform),
which is then executed deterministically. The LLM decides *what* to chart;
the plotting is plain code.

## Quickstart

```bash
cd chartlab
uv sync
uv run streamlit run src/chartlab/ui/app.py     # the app
uv run chartlab chart "compare AAPL and MSFT this year"   # CLI
```

## Demo vs. live

- **Demo mode** — a heuristic stand-in for the LLM parses your request,
  and charts are drawn from bundled synthetic price series. No API key,
  no network. Fully works in the hosted app.
- **Live mode** — your Anthropic or OpenAI key powers the real LLM parse,
  and prices are fetched live from Yahoo Finance (`yfinance`).

## Architecture

| Module | Responsibility |
|---|---|
| `models.py` | `ChartSpec`, `Series`, `PricePoint`, `Transform` |
| `llm.py`    | Multi-provider chat (Anthropic + OpenAI) |
| `nlp.py`    | Natural-language request → `ChartSpec` |
| `data.py`   | Price fetch via `yfinance` |
| `chart.py`  | Transforms (raw / indexed / % change) + Plotly figure |
| `mock.py`   | Heuristic demo parser + synthetic price series |

## Status

v0.1 — alpha. The demo-mode parser is a keyword heuristic; live mode uses
a real LLM. Bundled price series are **synthetic** — clearly labelled in
the UI. A portfolio project, not financial advice.
