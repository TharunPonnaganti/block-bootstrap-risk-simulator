# Block-Bootstrap Portfolio Risk Simulator

[![QA Checks](https://github.com/TharunPonnaganti/block-bootstrap-risk-simulator/actions/workflows/qa.yml/badge.svg)](https://github.com/TharunPonnaganti/block-bootstrap-risk-simulator/actions/workflows/qa.yml)

A risk simulator that shows you **the full range of possible outcomes** for your portfolio by replaying real historical return patterns -- then **checks its own forecasts** against what actually happened.

> **Not investment advice.** No trading or order capability of any kind.
> All examples use public tickers. All figures are nominal (not inflation-adjusted).
> Outputs are probabilistic ranges from historical data -- never predictions.

---

## What It Does

You give it a ticker (like `SPY` or `VTI`), a portfolio (`VTI:0.8, QQQ:0.2`), or a CSV from your brokerage. It runs thousands of simulated futures by reshuffling blocks of real past returns, and tells you:

- **How likely you are to make money** (P(profit)) over 1, 3, and 5 years
- **How bad the worst case looks** (VaR and CVaR -- the left-tail risk)
- **The biggest dip you'd sit through** (drawdown distribution)
- **The full range of outcomes** (P10 to P90 percentile cone)
- **How monthly investing (DCA/SIP) compares** to a one-time lump sum

It also has a **calibration module** that scores these forecasts against real history using walk-forward backtesting, and reports the results honestly -- including when the model shows no edge over a simple baseline.

---

## Why Block Bootstrap Instead of Normal Monte Carlo

Most portfolio simulators draw random returns from a bell curve (Normal distribution). That has two problems:

1. **It hides the risk you care about.** Real markets have fat tails, crash sequences, and volatility clustering. A bell curve throws all of that away and **understates how bad things can get**.

2. **Nobody checks if the probabilities are real.** A forecast of "72% chance of profit" sounds precise, but is it actually right?

This tool addresses both:

| | Normal Monte Carlo | **Block Bootstrap (this tool)** |
|---|---|---|
| Crash days | Nearly impossible (thin tails) | **Replayed from real history** |
| Volatility clustering | None (each day independent) | **Preserved within blocks** |
| Cross-asset correlation | Assumed via covariance matrix | **Preserved by joint resampling** |
| Distribution assumption | Strong (Normal or Student-t) | **None -- resamples real data** |

The method is the **circular moving-block bootstrap** (Politis & Romano, 1992): take contiguous blocks (~21 trading days) of real returns, stitch them into thousands of synthetic futures. Because the raw material is real, the simulation inherits the asset's genuine fat tails, volatility clustering, and drift.

---

## Supported Markets

| Market | Tickers | Currency | Risk-free rate | Example |
|---|---|---|---|---|
| **US** (default) | Any NYSE/NASDAQ symbol | `$` (USD) | 4.0% | `SPY`, `VTI`, `AAPL` |
| **India -- NSE** | Append `.NS` | `₹` (INR) | 6.5% | `NIFTYBEES.NS`, `RELIANCE.NS` |
| **India -- BSE** | Append `.BO` | `₹` (INR) | 6.5% | `NIFTYBEES.BO`, `RELIANCE.BO` |

Currency and cash benchmark rate are auto-detected from the ticker suffix.

---

## Installation

### What You Need

- **Python 3.10+** (tested on 3.13)
- **pip** (comes with Python)
- **Git** (to clone the repo)

### Setup

```bash
# 1. Clone
git clone https://github.com/TharunPonnaganti/block-bootstrap-risk-simulator.git
cd block-bootstrap-risk-simulator

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify -- should print 22/22 checks passed
python qa_check.py

# 5. Run
python stock_probability_engine.py SPY          # CLI
streamlit run app.py                            # Dashboard (opens in browser)
```

No API keys needed. Data is fetched from Yahoo Finance using Python's built-in `urllib`.

---

## Quick Start

### CLI

```bash
# US market
python stock_probability_engine.py                                # VTI (US total market, default)
python stock_probability_engine.py SPY                            # S&P 500
python stock_probability_engine.py SPY --blend                    # mix eras (recent + old crashes)
python stock_probability_engine.py SPY --blend --dca 500          # + $500/month recurring
python stock_probability_engine.py --portfolio "VTI:0.8,QQQ:0.2"  # multi-asset portfolio

# Indian market
python stock_probability_engine.py NIFTYBEES.NS                   # Nifty 50 ETF
python stock_probability_engine.py NIFTYBEES.NS --dca 5000        # + Rs.5,000/month SIP

# Machine-readable output
python stock_probability_engine.py SPY --json
```

### Sample Output

```
============================================================================
BLOCK-BOOTSTRAP PORTFOLIO RISK SIMULATOR
============================================================================
Data source      : Yahoo daily adj-close | SPY
History          : 1993-01-29 -> 2026-06-19  (33.4 yrs, 8419 obs)
Annualized drift :  10.0%   (geometric, full history)
Annualized vol   :  19.1%
Worst single day : -11.0%
Max drawdown seen: -55.2%   (actual peak->trough in history)
Invest amount    : $10,000   |   Method: circular block bootstrap, block=21, paths=10,000
Sampling window  : BLENDED
  window composition feeding the simulation:
     - 5y     weight  40%  |  1259 obs | drift +14.3% | vol  17%
     - 15y    weight  35%  |  3773 obs | drift +12.8% | vol  20%
     - full   weight  25%  |  8418 obs | drift +10.0% | vol  19%

############################################################################
#  HOLD 1 YEAR(S)        P(profit) 78.5%  (ABOVE the 70% threshold)
############################################################################
  Probability of profit          :  78.5%
  Probability of beating cash@4% :  71.0%
  Value of $10,000 after 1y:
      pessimistic  P10 : $      8,952   (-10%)
      median       P50 : $     11,343   (+13%)
      optimistic   P90 : $     14,005   (+40%)
  Annualized return (CAGR)       : P10 -10.5% | P50 +13.4% | P90 +40.1%
  Worst 5% of outcomes (VaR)     : end <= -19%   (CVaR, avg of that tail: -26%)
  Drawdown along the way         : median -13% | bad-case (worst 5%) -30%

############################################################################
#  HOLD 5 YEAR(S)        P(profit) 94.6%  (ABOVE the 70% threshold)
############################################################################
  Probability of profit          :  94.6%
  Probability of beating cash@4% :  85.5%
  Value of $10,000 after 5y:
      pessimistic  P10 : $     10,929   (+9%)
      median       P50 : $     17,932   (+79%)
      optimistic   P90 : $     29,459   (+195%)

============================================================================
MONTHLY RECURRING (SIP / DCA)  --  $500/month
============================================================================
############################################################################
#  1Y RECURRING   13 contributions   total invested $6,500   P(profit) 75.3%
############################################################################
  Probability of profit          :  75.3%   (portfolio > $6,500 total invested)
  Portfolio value after 1y:
      pessimistic  P10 : $      5,923   (-9% on invested)
      median       P50 : $      6,869   (+6% on invested)
      optimistic   P90 : $      7,857   (+21% on invested)
```

### Dashboard

```bash
streamlit run app.py
```

The dashboard includes:
- **Market toggle** (US / India) with pre-built ticker lists
- **Single ticker, portfolio, or CSV upload** modes
- **Monthly recurring (SIP/DCA)** with side-by-side comparison to lump sum
- **Interactive probability cones** -- lump sum (green) and recurring (blue)
- **Live threshold slider** -- simulation is cached, only the indicator updates
- **Currency-aware** -- auto-detects `$` or `₹`

---

## CLI Options

| Flag | Default | What it does |
|---|---|---|
| `--portfolio` | -- | Multi-asset allocation, e.g. `"VTI:0.8,QQQ:0.2"` (cross-asset correlation preserved) |
| `--csv` | -- | Use a local CSV instead of Yahoo (works with Fidelity, Robinhood, Schwab exports) |
| `--dca` | -- | Monthly recurring contribution (SIP/DCA), shown alongside lump sum |
| `--blend` | off | Mix return blocks across time periods (5y/15y/full) to avoid being locked into one era |
| `--years` | full | Use only the last N years of history |
| `--threshold` | 0.70 | P(profit) level for the ABOVE/BELOW indicator |
| `--amount` | 10000 | Hypothetical lump-sum amount |
| `--paths` | 10000 | Number of simulated futures (more = smoother, slower) |
| `--haircut` | 0.0 | Remove this fraction of historical drift (0-1) as a stress test |
| `--json` | off | Output machine-readable JSON |

---

## How It Works

### Core Engine (`stock_probability_engine.py`)

**Data layer** -- uses only Python's built-in `urllib`, `json`, and `csv` modules. Pulls dividend/split-adjusted daily prices from Yahoo Finance. Also reads any brokerage CSV (auto-detects date and price columns). No API key needed.

**Three bootstrap modes:**

1. **Single-series** -- Circular block bootstrap on one ticker's log-returns. Default is `VTI` (US total market). Single stocks work but get a `WEAKER PRIOR` warning because one company's past is idiosyncratic.

2. **Era-blended (`--blend`)** -- Mixes blocks across time windows (5y at 40%, 15y at 35%, full history at 25%). A single simulated path can stitch a calm 2017 stretch onto a 2008-style crash block, so the future isn't locked into one regime.

3. **Portfolio (`--portfolio`)** -- Multiple assets are date-aligned, and **one set of block indices is applied to every asset at once**. Each resampled time step is a real historical cross-section, which preserves the actual cross-asset correlation -- no covariance matrix or Gaussian copula needed.

**What you get per horizon (1, 3, 5 years):**
- Percentile cone (P5 through P95) of portfolio value
- `P(profit)` -- fraction of simulated futures ending above what you invested
- `P(beats cash)` -- fraction beating a risk-free cash benchmark
- **VaR / CVaR (Expected Shortfall)** -- how bad the worst 5% of outcomes look
- **Drawdown distribution** -- the biggest peak-to-trough dip across all paths
- **DCA/SIP results** (if enabled) -- same market scenarios, different cash-flow timing

**Stress testing:** `--haircut` removes a fraction of historical drift to see what happens if the future is less rosy than the past.

### Walk-Forward Calibration (`calibration.py`)

The bootstrap produces internally consistent probabilities (validated by 22 invariant tests). But that doesn't tell you whether a "72% chance of profit" is actually right. This module checks by **stepping through history**:

1. At each point in time, fit the bootstrap using **only data available up to that point**
2. Forecast P(profit) and outcome percentiles for the next H years
3. Record what actually happened
4. Score the (forecast, outcome) pairs

**Scoring:**
- **Brier Score** -- how far off the probability forecasts were (lower = better)
- **Brier Skill Score (BSS)** -- whether the model beats a naive baseline that always predicts the historical average. BSS > 0 means it adds value; BSS <= 0 means it doesn't
- **Reliability curve** -- predicted probability vs. what actually happened, bucketed into deciles. A good model sits on the diagonal
- **PIT coverage** -- checks whether the whole predicted distribution is well-calibrated, not just P(profit)

**Honesty rules:**
- The baseline at each point uses **only past data** (no peeking at the future)
- Reports both raw sample count and effective independent N (overlapping windows are correlated)
- Origins with too little training data are excluded

### QA Suite (`qa_check.py`)

22 invariant tests across three areas:

- **Engine (12 tests):** determinism, unbiased drift, drawdown sign, scale invariance, VaR/CVaR ordering, CAGR consistency, monotonic P(profit), stress tests, error handling, robustness
- **Portfolio (3 tests):** weights sum to 1, joint resampling preserves correlation, independent resampling destroys it
- **Calibration (6 tests):** perfect forecast scores 0, coin-flip scores 0.25, base-rate identity, BSS=0 for naive model, informative forecaster beats baseline, bucket counts sum to N

---

## Calibration Results -- SPY, 1993-2026

![SPY 1-year reliability curve](reliability_curve.png)

| Horizon | Brier (model) | Brier (baseline) | **BSS** | Independent N |
|---|---|---|---|---|
| 1 year | 0.186 | 0.186 | **-0.00** | ~28 |
| 3 years | 0.032 | 0.063 | **+0.49** | ~6 |

**What this means:**

- **1-year:** The model is tied with a simple baseline (BSS near 0). It doesn't beat naive forecasting, but it doesn't lose either. With only ~28 independent windows, there isn't enough data to tell.

- **3-year:** The +0.49 looks good, but it's based on only ~6 independent windows, all of which were profitable. This is a small-sample artifact, not real predictive skill.

- **PIT (distribution check):** At 1 year, outcomes fall below the forecast P10/P50/P90 at rates of 15%/42%/96% (targets: 10/50/90). The distribution shape is reasonable but slightly optimistic at the median.

**Bottom line:** P(profit) is best used as a **risk-distribution indicator**, not a directional signal. The real value of this tool is the **shape of the distribution** -- drawdown, VaR/CVaR, and the percentile cone.

---

## Known Limitations

1. **The past isn't the future.** The bootstrap samples from historical returns. A structurally new regime (different interest rates, new market structure) can't be sampled because it hasn't happened yet.

2. **Blocks are stitched independently.** This means multi-month trends aren't reproduced across block boundaries. Sustained crashes involve momentum that independent blocks fragment, so multi-year tail risk may be **optimistic**.

3. **Single stocks have weak priors.** One company's history is dominated by idiosyncratic risk (earnings surprises, management changes). The tool warns about this and steers toward index funds or portfolios.

4. **All values are nominal.** No inflation adjustment. At long horizons, real purchasing power can differ significantly.

5. **Small calibration sample.** ~33 years of US equity data gives only ~28 independent 1-year windows and ~6 independent 5-year windows. This fundamentally limits how precisely any model can be validated at long horizons.

---

## Project Structure

| File | What it does |
|---|---|
| `stock_probability_engine.py` | Core engine: data fetching (Yahoo/CSV), circular block bootstrap (single + blended + portfolio), DCA/SIP simulation, risk metrics (VaR/CVaR/drawdown), CLI output |
| `calibration.py` | Walk-forward backtest: expanding-window calibration, Brier Score, Brier Skill Score, reliability curve, PIT coverage |
| `qa_check.py` | 22 statistical tests: engine determinism, scale invariance, tail-risk ordering, portfolio correlation, calibration math |
| `app.py` | Streamlit dashboard: lump sum vs. recurring comparison, probability cones, risk cards, market toggle |
| `requirements.txt` | Python dependencies (numpy, matplotlib, streamlit, pandas, altair) |
| `reliability_curve.png` | Sample calibration output (SPY, 1-year horizon) |

---

## Tech Stack

- **Python 3.10+** -- tested on 3.13
- **NumPy** -- all bootstrap and statistical computations
- **matplotlib** -- calibration reliability curve
- **Streamlit + Altair** -- interactive local dashboard
- **Standard library only** for data fetching (`urllib`, `json`, `csv`)

---

## Disclaimer

This is a **research and analysis tool**, not financial advice and not a recommendation to buy or sell any security. It has **no trading or order-execution capability**. Probabilities come from historical data that may not represent the future. The model shows no demonstrated directional skill versus a baseline. Do your own due diligence.
