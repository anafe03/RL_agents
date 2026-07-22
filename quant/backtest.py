"""
Your first backtest: a moving-average crossover on SPY.

The point of this file is NOT to make money. It's to learn the loop every quant
lives in:  get data → define a rule → simulate it honestly → measure it → see
why it (probably) loses to just buying and holding.

Run it:
    cd quant
    python3 backtest.py                 # default: SMA 50/200 on SPY
    python3 backtest.py 20 100          # try a faster/slower pair

Only needs pandas + numpy (already installed). Pulls free daily data from
Stooq; if you're offline it falls back to a synthetic series so it always runs.
"""

from __future__ import annotations
import sys
import io
import urllib.request
import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_PER_TRADE = 0.0005  # 5 basis points each time we get in or out (fees+slippage)


# ----------------------------------------------------------------------------
# 1. DATA  — real if we can reach Stooq, synthetic otherwise (so it always runs)
# ----------------------------------------------------------------------------
def get_prices(symbol: str = "spy.us") -> tuple[pd.Series, str]:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        if "Close" not in df or len(df) < 300:
            raise ValueError("bad data")
        df["Date"] = pd.to_datetime(df["Date"])
        s = df.set_index("Date")["Close"].astype(float).sort_index()
        return s, f"real SPY daily closes from Stooq ({len(s)} days)"
    except Exception:
        # geometric brownian motion fallback — realistic-looking, but NOT a real
        # market, so treat any "profit" here as meaningless.
        rng = np.random.default_rng(7)
        n = 2500
        daily = rng.normal(0.0005, 0.011, n)  # ~13%/yr drift, ~17%/yr vol
        price = 100 * np.exp(np.cumsum(daily))
        idx = pd.bdate_range("2015-01-01", periods=n)
        return pd.Series(price, index=idx), f"SYNTHETIC data ({n} days) — offline fallback"


# ----------------------------------------------------------------------------
# 2. STRATEGY  — long when fast SMA is above slow SMA, else flat (in cash)
# ----------------------------------------------------------------------------
def sma_crossover(price: pd.Series, fast: int, slow: int) -> pd.Series:
    f = price.rolling(fast).mean()
    s = price.rolling(slow).mean()
    # position we WANT to hold, decided from data available at close of day t
    position = (f > s).astype(int)
    # ...but we can only act NEXT day. Shifting by 1 prevents "lookahead bias" —
    # the #1 way beginners fool themselves into fake profits.
    return position.shift(1).fillna(0)


# ----------------------------------------------------------------------------
# 3. SIMULATE + 4. MEASURE
# ----------------------------------------------------------------------------
def metrics(equity: pd.Series, rets: pd.Series) -> dict:
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(rets) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    return {"total": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def run(price: pd.Series, fast: int, slow: int) -> None:
    daily = price.pct_change().fillna(0)
    pos = sma_crossover(price, fast, slow)

    trades = pos.diff().abs().fillna(0)   # 1 each time we enter or exit
    cost = trades * COST_PER_TRADE
    strat_daily = pos * daily - cost       # strategy's daily return, after costs
    n_trades = int(trades.sum())
    exposure = pos.mean()                  # fraction of time actually invested

    strat_equity = (1 + strat_daily).cumprod()
    hold_equity = (1 + daily).cumprod()

    s = metrics(strat_equity, strat_daily)
    h = metrics(hold_equity, daily)

    def row(name, m, extra=""):
        print(f"  {name:<16} {m['total']*100:8.1f}% {m['cagr']*100:7.1f}% "
              f"{m['vol']*100:7.1f}% {m['sharpe']:7.2f} {m['max_dd']*100:8.1f}%   {extra}")

    print("\n" + "=" * 72)
    print(f"  SMA {fast}/{slow} crossover   vs   buy & hold")
    print("=" * 72)
    print(f"  {'':<16} {'total':>8} {'CAGR':>7} {'vol':>7} {'Sharpe':>7} {'maxDD':>8}")
    print("-" * 72)
    row("strategy", s, f"{n_trades} trades, {exposure*100:.0f}% invested")
    row("buy & hold", h, "always invested")
    print("-" * 72)

    verdict = "BEAT" if s["total"] > h["total"] else "LOST TO"
    print(f"\n  → the strategy {verdict} buy & hold on total return.")
    print("  Read it like a quant:")
    print("   • Sharpe (return per unit of risk) matters more than raw return.")
    print("   • Crossover sits in cash a lot, so it usually has a SMALLER drawdown")
    print("     but also gives up upside during big rallies.")
    print("   • Costs are real: every trade above bleeds 5 bps. More trades ≠ better.")
    print("   • Most simple rules LOSE to buy & hold. Finding the rare one that")
    print("     doesn't — after costs, out of sample — is the actual job.\n")


if __name__ == "__main__":
    fast = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    slow = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    price, note = get_prices()
    print(f"\ndata: {note}")
    run(price, fast, slow)
