# quant — your learning lab

The fastest way into quant isn't more reading — it's running one honest
backtest, watching it fail, and understanding why. This is that.

## Run it
```bash
cd quant
python3 backtest.py            # SMA 50/200 crossover on SPY vs buy & hold
python3 backtest.py 20 100     # try any fast/slow pair
```
Only needs **pandas + numpy** (already installed). It pulls free daily SPY data
from Stooq automatically; if you're offline it falls back to a synthetic series
so it always runs. *(Synthetic results are meaningless — the numbers only mean
something on real data.)*

## What it teaches
The four steps every strategy goes through, done honestly:
1. **Data** — real prices.
2. **Rule** — long when the fast moving average is above the slow one, else cash.
3. **Simulate** — with the signal **shifted one day** (no lookahead — the #1 way
   beginners fake profits) and **5 bps cost per trade** (fees + slippage).
4. **Measure** — total return, CAGR, volatility, **Sharpe**, **max drawdown**,
   trade count, time invested — all vs just buying and holding.

## The lesson (on real SPY)
The crossover usually **loses to buy & hold on total return** but has a **smaller
drawdown** — it sidesteps some crashes but also misses upside. That trade-off,
read through Sharpe and drawdown rather than raw return, is how a quant actually
judges a strategy. Most simple rules lose after costs; finding the rare one that
doesn't — *out of sample, after costs* — is the whole job.

## The one trap to internalize
Your ML background is an **edge** (you understand overfitting) and a **trap**
(financial data is non-stationary and low-signal, so it's trivially easy to fool
yourself with an in-sample fit). Read **López de Prado, _Advances in Financial
Machine Learning_** first, for exactly this reason.

## Next steps to extend this
- Split data into **train / test** and only trust out-of-sample results.
- Add strategies: mean-reversion (RSI), momentum, pairs trading.
- Swap the hand-rolled engine for **`backtesting.py`** or **`vectorbt`** once you
  want speed + parameter sweeps (small installs — mind the disk).
- Pull data with **`yfinance`** or a CSV if Stooq is flaky.
