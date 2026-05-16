"""Streamlit UI for chartlab.

Run locally:
    cd chartlab
    uv sync
    uv run streamlit run src/chartlab/ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from chartlab import llm
from chartlab.chart import build_figure, summarize
from chartlab.mock import DEMO_TICKERS, demo_series, make_mock_chat
from chartlab.nlp import parse_request

st.set_page_config(page_title="chartlab", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#3fb950"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0e1117; }}
    .cl-header {{
        background: linear-gradient(135deg, #161b22 0%, #0e1117 100%);
        border: 1px solid #21262d;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .cl-header .title {{
        font-size: 1.7rem; font-weight: 800; color: {ACCENT};
        font-family: -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.02em;
    }}
    .cl-header .sub {{ color: #8b949e; font-size: 0.88rem; margin-top: 0.2rem; }}
    .cl-chip {{
        display: inline-block; font-family: "SF Mono", Menlo, monospace;
        font-size: 0.78rem; padding: 0.18rem 0.6rem; border-radius: 5px;
        margin-right: 0.35rem; background: #21262d; color: #c9d1d9;
    }}
    .cl-chip b {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/chartlab"
EXAMPLES = [
    "Compare Apple and Microsoft over the last year",
    "How has Nvidia done this year, in percent",
    "Bitcoin vs the S&P 500, indexed to 100",
    "Tesla price over the past 6 months",
]

if "request" not in st.session_state:
    st.session_state.request = EXAMPLES[0]
if "result" not in st.session_state:
    st.session_state.result = None


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 📈 chartlab")
    st.caption("Describe a chart in plain English — an LLM builds it.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Demo", "Live (your API key)"],
        help="Demo uses a keyword parser + synthetic prices — no key, no "
        "network. Live uses a real LLM and fetches prices from Yahoo Finance.",
    )
    model = "claude-sonnet-4-6"
    api_key = ""
    if mode.startswith("Live"):
        model = st.selectbox(
            "Model", llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
            help="Provider auto-detected by prefix. gpt-4o-mini is cheapest.",
        )
        key_var = "OPENAI_API_KEY" if llm._is_openai_model(model) else "ANTHROPIC_API_KEY"
        api_key = st.text_input(key_var, type="password",
                                help="Used only in your browser session.")
        st.caption("Live mode fetches real prices from Yahoo Finance.")
    else:
        st.caption("Demo prices are **synthetic** — shapes are illustrative, "
                   "not real market data.")

    st.markdown("---")
    st.markdown("**Try an example:**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.request = ex
            st.rerun()

    st.markdown("---")
    st.caption(f"Demo tickers: {', '.join(DEMO_TICKERS)}")
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    """
    <div class="cl-header">
        <div class="title">chartlab</div>
        <div class="sub">Natural-language financial charting — describe what
        you want, an LLM turns it into a chart spec, chartlab plots it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- request ----------------------------------------------------------------

request = st.text_input(
    "Describe the chart you want",
    key="request",
    placeholder="e.g. compare Apple and Microsoft over the last year",
)
go = st.button("📈 Chart it", type="primary", use_container_width=True)

if go:
    if not request.strip():
        st.warning("Type a request first — or pick an example from the sidebar.")
    elif mode.startswith("Live") and not api_key:
        st.warning(f"Live mode needs an API key. Enter one in the sidebar, "
                   f"or switch to Demo mode.")
    else:
        if mode == "Demo":
            llm.set_chat_fn(make_mock_chat())
        else:
            os.environ[key_var] = api_key
            llm.reset_chat_fn()
        try:
            with st.spinner("Parsing your request..."):
                spec, cost = parse_request(request, model=model)
            series_list = []
            failed = []
            if mode == "Demo":
                series_list = [demo_series(t, spec.period) for t in spec.tickers]
            else:
                from chartlab.data import fetch_all
                with st.spinner("Fetching prices from Yahoo Finance..."):
                    series_list, failed = fetch_all(spec.tickers, spec.period)
            st.session_state.result = {
                "spec": spec, "series": series_list, "failed": failed,
                "cost": cost, "mode": mode,
            }
        except Exception as e:  # noqa: BLE001
            st.session_state.result = None
            st.error(f"Couldn't build that chart — {type(e).__name__}: {e}")
        finally:
            llm.reset_chat_fn()


# --- result -----------------------------------------------------------------

result = st.session_state.result
if result is not None:
    spec = result["spec"]
    series_list = result["series"]

    st.markdown(
        f'<div style="margin:0.4rem 0 0.8rem 0">'
        f'<span class="cl-chip">tickers <b>{", ".join(spec.tickers) or "—"}</b></span>'
        f'<span class="cl-chip">period <b>{spec.period}</b></span>'
        f'<span class="cl-chip">view <b>{spec.transform.value}</b></span>'
        f'<span class="cl-chip">parsed by <b>'
        f'{"heuristic (demo)" if result["mode"] == "Demo" else "LLM"}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if result["cost"]:
        st.caption(f"LLM parse cost: ${result['cost']:.5f}")
    if result["failed"]:
        st.warning("Couldn't fetch: " + ", ".join(result["failed"])
                   + " — unknown symbol, or Yahoo rate-limited the request.")

    if not series_list:
        st.error("No data to chart. Check the tickers and try again.")
    else:
        st.plotly_chart(build_figure(spec, series_list), use_container_width=True)

        cols = st.columns(len(series_list))
        for col, series in zip(cols, series_list):
            s = summarize(series, spec.transform)
            col.metric(
                series.ticker,
                f"${s['end']:,.2f}",
                f"{s['change_pct']:+.2f}%",
            )

    if result["mode"] == "Demo":
        st.caption("Demo mode — synthetic prices. Switch to Live mode with an "
                   "API key for a real LLM parse and live Yahoo Finance data.")
else:
    st.info("Describe a chart above and hit **Chart it** — or pick an example "
            "from the sidebar.")
