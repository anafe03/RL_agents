"""
Interactive backtest demo (Streamlit).

Deploys to Streamlit Community Cloud → a public https://<name>.streamlit.app
URL you can link from your website. Runs in the cloud, not on your machine.

Local preview (optional):  pip install streamlit && streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from backtest import get_prices, sma_crossover, metrics

st.set_page_config(page_title="Backtest Lab", page_icon="📈", layout="wide")

st.title("📈 Backtest Lab")
st.caption(
    "An **honest** moving-average crossover vs. buy & hold — lookahead-free "
    "signals, real trading costs, and the metrics a quant actually reads."
)

# ---- controls ----
with st.sidebar:
    st.header("Strategy")
    symbol = st.text_input("Symbol (Stooq code)", "spy.us", help="e.g. spy.us, aapl.us, qqq.us")
    fast = st.slider("Fast SMA (days)", 5, 120, 50)
    slow = st.slider("Slow SMA (days)", 20, 300, 200)
    cost_bps = st.slider("Cost per trade (bps)", 0, 25, 5,
                         help="fees + slippage, charged each time you enter or exit")
    st.markdown("---")
    st.caption("Long when the fast average is above the slow one, otherwise in cash.")

if fast >= slow:
    st.warning("Fast SMA must be **shorter** than slow SMA. Adjust the sliders.")
    st.stop()


@st.cache_data(show_spinner=True)
def load(sym: str):
    return get_prices(sym)


price, note = load(symbol)
if "SYNTHETIC" in note:
    st.info(f"⚠️ {note} — couldn't reach live data, so these numbers are illustrative only.")
else:
    st.success(f"✓ {note}")

# ---- simulate (mirrors backtest.py, cost from the slider) ----
daily = price.pct_change().fillna(0)
pos = sma_crossover(price, fast, slow)
trades = pos.diff().abs().fillna(0)
cost = trades * (cost_bps / 10_000)
strat_daily = pos * daily - cost
strat_eq = (1 + strat_daily).cumprod()
hold_eq = (1 + daily).cumprod()

s = metrics(strat_eq, strat_daily)
h = metrics(hold_eq, daily)
n_trades = int(trades.sum())
exposure = float(pos.mean())

# ---- headline metrics ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total return", f"{s['total']*100:.1f}%", f"{(s['total']-h['total'])*100:+.1f}% vs hold")
c2.metric("Sharpe", f"{s['sharpe']:.2f}", f"{s['sharpe']-h['sharpe']:+.2f} vs hold")
c3.metric("Max drawdown", f"{s['max_dd']*100:.1f}%", f"{(s['max_dd']-h['max_dd'])*100:+.1f}% vs hold",
          delta_color="inverse")
c4.metric("Trades", f"{n_trades}", f"{exposure*100:.0f}% invested")

# ---- equity curve ----
st.subheader("Growth of $1")
eq = pd.DataFrame({"strategy": strat_eq, "buy & hold": hold_eq})
st.line_chart(eq, height=340)

# ---- drawdown ----
st.subheader("Drawdown")
dd = pd.DataFrame({
    "strategy": strat_eq / strat_eq.cummax() - 1,
    "buy & hold": hold_eq / hold_eq.cummax() - 1,
})
st.area_chart(dd, height=220)

# ---- full table ----
with st.expander("All metrics"):
    tbl = pd.DataFrame({
        "strategy": [s["total"], s["cagr"], s["vol"], s["sharpe"], s["max_dd"]],
        "buy & hold": [h["total"], h["cagr"], h["vol"], h["sharpe"], h["max_dd"]],
    }, index=["Total return", "CAGR", "Volatility", "Sharpe", "Max drawdown"])
    st.dataframe(
        tbl.style.format({"strategy": "{:.3f}", "buy & hold": "{:.3f}"}),
        use_container_width=True,
    )

# ---- the lesson ----
verdict = "**beat**" if s["total"] > h["total"] else "**lost to**"
st.markdown(
    f"""
    ### How to read this
    On this run the strategy {verdict} buy & hold on total return — but that's
    the *least* important line. What matters:
    - **Sharpe** (return per unit of risk) is the real scorecard, not raw return.
    - The crossover sits in cash {(1-exposure)*100:.0f}% of the time, so it usually
      has a **smaller drawdown** but gives up upside in big rallies.
    - Every trade costs {cost_bps} bps — **more trades ≠ better.**
    - Tuning the sliders until it looks great is **overfitting.** The honest test
      is out-of-sample, after costs. Most simple rules lose; finding the rare one
      that doesn't is the whole job.
    """
)
st.caption("Source: hand-rolled backtest (pandas/numpy), lookahead-free, costs included.")
