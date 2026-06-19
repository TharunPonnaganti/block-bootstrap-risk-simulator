"""
BLOCK-BOOTSTRAP PORTFOLIO RISK SIMULATOR  (core engine)
=======================================================
Risk-and-return distribution for a diversified INDEX or multi-asset PORTFOLIO
(primary use), or a single stock (secondary, weaker prior): "given the real
historical return distribution, what is the range of outcomes over horizon H?"
Reports percentile outcomes, P(profit), VaR/CVaR, drawdown -- and a neutral
P(profit)-vs-threshold INDICATOR (not a buy/sell signal) from a threshold YOU set.

  PRIMARY  : a diversified fund (default VTI) or --portfolio "VTI:0.8,QQQ:0.2"
             (multi-asset, with cross-asset correlation preserved by joint
             resampling). The bootstrap prior is statistically sound here.
  SECONDARY: a single stock -- supported, but flagged WEAKER PRIOR because one
             name is idiosyncratic.
  Forecast quality is validated out-of-sample in calibration.py (walk-forward
  reliability, Brier score, skill vs a naive base rate).

METHOD (kept deliberately rigorous -- not a flat-average projection):
  Circular BLOCK BOOTSTRAP of the asset's OWN historical returns. We resample
  contiguous blocks of real past returns and stitch them into thousands of
  synthetic future paths. Because we reuse REAL return sequences, the
  simulation inherits the actual:
     - fat tails / crash days     (a normal-curve Monte Carlo erases these)
     - volatility clustering       (calm and storm cluster, as in reality)
     - drift                       (its real historical trend)
  This is closer to how a desk stress-tests a position than mu/sigma guesses.

P(profit)-vs-THRESHOLD INDICATOR:
  At each horizon we report P(end value > amount invested) and flag whether it is
  ABOVE or BELOW a threshold YOU set. This is a mechanical comparison, NOT a
  buy/sell signal -- walk-forward calibration (calibration.py) shows P(profit) has
  no demonstrated skill versus a base-rate benchmark, so it must not be read as a
  trade recommendation. NOT investment advice, NOT a prediction.

HONESTY NOTES:
  - A single stock is idiosyncratic: its past distribution is a WEAKER prior
    than a diversified index. The script reports how much history it had and
    warns when that history is short or spans only a bull market.
  - All figures are NOMINAL (not inflation-adjusted).
  - Bootstrap drift = the stock's realized historical drift. If that history was
    an unusual boom, the odds shown will be too rosy. DRIFT_HAIRCUT lets you
    shave the drift to stress a more conservative view.

DATA:
  Default: pulls public, dividend/split-adjusted daily history from Yahoo by
  ticker. Override with your own CSV (Fidelity / Robinhood / anywhere) -- the
  parser auto-detects a date column and a price column ('Adj Close' preferred).

USAGE:
  python stock_probability_engine.py                              # default index VTI, full history
  python stock_probability_engine.py SPY                          # any diversified fund
  python stock_probability_engine.py --portfolio "VTI:0.8,QQQ:0.2"  # multi-asset portfolio
  python stock_probability_engine.py AAPL                         # single stock (weaker-prior caveat)
  python stock_probability_engine.py VTI --years 15               # cap to last 15 years
  python stock_probability_engine.py VTI --blend                  # blend eras (see BLEND)
  python stock_probability_engine.py --csv mydata.csv             # your own export
  python stock_probability_engine.py SPY --json                   # machine-readable output (for an app)
  extra knobs: --threshold 0.7  --amount 10000  --paths 10000  --haircut 0.25
"""

import sys
import json
import csv as csvmod
import datetime as dt
import urllib.request
import numpy as np

# ----------------------------------------------------------------------
# 1. INPUTS  --  CHANGE THESE
# ----------------------------------------------------------------------
TICKER          = "VTI"       # PRIMARY use = a diversified index/fund (bootstrap prior is sound here)
PORTFOLIO       = None        # e.g. "VTI:0.8,QQQ:0.2" -> multi-asset, correlation-preserving bootstrap
CSV_PATH        = None        # e.g. r"C:\Users\you\Downloads\VTI.csv" to use your own export

# Recognized broad/diversified funds: a single one of these is a legitimate prior.
# Any OTHER single ticker is treated as a single stock and earns a WEAKER-PRIOR caveat.
DIVERSIFIED = {"VTI", "VOO", "SPY", "IVV", "ITOT", "SCHB", "SCHX", "QQQ", "DIA", "IWM",
               "VXUS", "VEU", "VEA", "VWO", "VT", "ACWI", "BND", "AGG", "BNDX",
               "VTV", "VUG", "VYM", "SCHD", "VIG", "RSP"}
AMOUNT          = 10_000.0    # lump sum hypothetically invested today (P(profit) is scale-free)
HORIZONS_YEARS  = [1, 3, 5]   # holding periods to report
PROFIT_THRESHOLD = 0.70       # indicator flags P(profit) ABOVE/BELOW this  <-- YOUR threshold
CASH_RATE       = 0.04        # annual risk-free / cash benchmark for "beats cash" odds
VAR_CONF        = 0.95        # confidence for VaR / CVaR (0.95 => worst 5% tail)
DRIFT_HAIRCUT   = 0.00        # 0 = use real historical drift; 1 = strip drift to zero (stress test)
N_PATHS         = 10_000      # number of bootstrap paths
BLOCK_DAYS      = 21          # bootstrap block length at DAILY frequency (~1 trading month)
HISTORY_YEARS   = None        # None = full available daily history; or an int to cap lookback (--years)
SEED            = 42

# Blended-window mode (--blend): instead of forcing one history window, mix
# return blocks across eras so paths span calm recent regimes AND old crashes.
# Keys = lookback years (None = full history), values = weights (auto-normalized).
# Each simulated month is drawn from one era according to these odds.
BLEND           = {5: 0.40, 15: 0.35, None: 0.25}

# ----------------------------------------------------------------------
# 2. DATA LAYER
# ----------------------------------------------------------------------
def fetch_yahoo(ticker, history_years=None):
    """Pull dividend/split-adjusted DAILY closes from Yahoo's public chart API
    using only the standard library. We pass explicit period1/period2 (not
    range=max, which Yahoo silently coarsens to quarterly bars) to guarantee
    true daily granularity. Returns (dates[list], prices[np.array], source)."""
    import time
    p2 = int(time.time())
    p1 = 0 if history_years is None else max(0, p2 - int(history_years * 365.25 * 86400))
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    j = json.loads(raw)
    if j.get("chart", {}).get("error"):
        raise RuntimeError(f"Yahoo error for '{ticker}': {j['chart']['error']}")
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    # prefer adjusted close; fall back to raw close
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        adj = res["indicators"]["quote"][0]["close"]
    dates, prices = [], []
    for t, p in zip(ts, adj):
        if p is None:
            continue
        dates.append(dt.date.fromtimestamp(t))
        prices.append(float(p))
    return dates, np.array(prices, dtype=float), f"Yahoo daily adj-close | {ticker.upper()}"


def parse_csv(path):
    """Auto-detect a date column and a price column in an arbitrary brokerage CSV."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csvmod.reader(fh))
    # find header row (first row containing something date-like)
    header = rows[0]
    cols = [c.strip().lower() for c in header]

    def find(cands):
        for cand in cands:
            for i, c in enumerate(cols):
                if cand == c:
                    return i
        for cand in cands:                       # loose contains-match fallback
            for i, c in enumerate(cols):
                if cand in c:
                    return i
        return None

    di = find(["date", "trade date", "as of date", "period"])
    pi = find(["adj close", "adjclose", "adj_close", "adjusted close",
               "close", "close/last", "price", "last price", "last", "value", "nav"])
    if di is None or pi is None:
        raise RuntimeError(f"Could not find date/price columns in header: {header}")

    recs = []
    for r in rows[1:]:
        if len(r) <= max(di, pi):
            continue
        ds, ps = r[di].strip(), r[pi].strip().replace("$", "").replace(",", "")
        if not ds or not ps:
            continue
        d = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%Y/%m/%d", "%b %d, %Y"):
            try:
                d = dt.datetime.strptime(ds, fmt).date(); break
            except ValueError:
                continue
        if d is None:
            continue
        try:
            p = float(ps)
        except ValueError:
            continue
        if p > 0:
            recs.append((d, p))
    if len(recs) < 30:
        raise RuntimeError(f"Only parsed {len(recs)} usable rows from {path}")
    recs.sort(key=lambda x: x[0])
    dates = [d for d, _ in recs]
    prices = np.array([p for _, p in recs], dtype=float)
    return dates, prices, f"CSV: {path}"


def load_prices(ticker, csv_path):
    """Load the FULL available series; per-window slicing is done in-engine so
    --years and --blend operate on the same loaded history."""
    if csv_path:
        return parse_csv(csv_path)
    return fetch_yahoo(ticker, None)


def parse_weights(spec):
    """Parse a portfolio spec into {TICKER: weight}.
    'VTI:0.8,QQQ:0.2' -> {'VTI':0.8,'QQQ':0.2};  'VTI,QQQ' -> equal-weight."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            t, w = part.split(":")
            out[t.strip().upper()] = float(w)
        else:
            out[part.strip().upper()] = None
    if any(w is None for w in out.values()):          # equal-weight if any unspecified
        eq = 1.0 / len(out)
        out = {t: eq for t in out}
    return out


def fetch_portfolio(weights, history_years=None):
    """Fetch each component, INNER-JOIN on common trading dates, and return the
    date-aligned SIMPLE-return matrix plus a normalized weight vector. Aligning on
    shared dates is what lets the joint bootstrap preserve cross-asset correlation.
    Returns (rdates, R_simple [n_obs x n_assets], weights_vec, tickers, source)."""
    tickers = list(weights.keys())
    series = {}
    for t in tickers:
        d, p, _ = fetch_yahoo(t, history_years)
        series[t] = dict(zip(d, p))
    common = sorted(set.intersection(*[set(series[t]) for t in tickers]))
    if len(common) < 60:
        raise RuntimeError(f"Only {len(common)} overlapping dates across {tickers}; need >= 60.")
    price = np.array([[series[t][d] for t in tickers] for d in common], dtype=float)
    R = price[1:] / price[:-1] - 1.0                  # aligned simple returns
    rdates = common[1:]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()
    src = "Yahoo daily adj-close | portfolio " + ", ".join(
        f"{t} {wi*100:.0f}%" for t, wi in zip(tickers, w))
    return rdates, R, w, tickers, src


# ----------------------------------------------------------------------
# 3. DIAGNOSTICS  --  how good is this prior?
# ----------------------------------------------------------------------
def historical_max_drawdown(prices):
    runmax = np.maximum.accumulate(prices)
    return float((prices / runmax - 1.0).min())


def detect_frequency(dates):
    """Return (periods_per_year, block_len) inferred from the date spacing."""
    days = np.array([(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)])
    med = float(np.median(days))
    if med <= 3:        # daily / business-daily
        return 252.0, BLOCK_DAYS
    if 5 <= med <= 10:  # weekly
        return 52.0, max(2, round(BLOCK_DAYS / 5))
    if 25 <= med <= 35: # monthly
        return 12.0, 6
    ppy = 365.25 / max(med, 1.0)
    return ppy, max(2, round(ppy / 12))


def slice_window(dates, log_ret, years):
    """Return the log-return sub-array for the last `years` (None = all).
    log_ret[i] corresponds to dates[i+1]."""
    rdates = dates[1:]
    if years is None:
        return rdates, log_ret
    cutoff = rdates[-1] - dt.timedelta(days=int(years * 365.25))
    mask = np.array([d >= cutoff for d in rdates])
    return [d for d, m in zip(rdates, mask) if m], log_ret[mask]


def annualize(returns, ppy):
    """Return (geometric annual drift, annualized vol) for a log-return array."""
    drift = float(np.expm1(returns.mean() * ppy))
    vol = float(returns.std(ddof=1) * np.sqrt(ppy)) if len(returns) > 1 else float("nan")
    return drift, vol


# ----------------------------------------------------------------------
# 4. BOOTSTRAP ENGINE
# ----------------------------------------------------------------------
def _block_indices(lengths, weights, n_periods, n_paths, block, rng):
    """Circular moving-block sampler of TIME indices into a pooled timeline of
    length sum(lengths). For each block we pick an era (window) by `weights`, then
    draw a contiguous block, wrapping circularly within that era. Returns an
    (n_paths, n_periods) int array of indices into the concatenated pool.

    Factored out so the single-series and multi-asset (portfolio) bootstraps share
    EXACTLY the same index draw. The portfolio case applies one index array to
    every asset column simultaneously -- that is precisely what preserves the real
    cross-asset correlation (each resampled timestep is a true historical row)."""
    lengths = np.asarray(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    p = np.asarray(weights, dtype=float); p = p / p.sum()
    nblocks = int(np.ceil(n_periods / block))
    wsel = rng.choice(len(lengths), size=(n_paths, nblocks), p=p)  # era per block slot
    Tsel = lengths[wsel]                                           # era length per slot
    Osel = offsets[wsel]                                           # era offset in pool
    start_local = (rng.random((n_paths, nblocks)) * Tsel).astype(np.int64)
    offs = np.arange(block)
    local = (start_local[:, :, None] + offs[None, None, :]) % Tsel[:, :, None]   # circular within era
    return (local + Osel[:, :, None]).reshape(n_paths, nblocks * block)[:, :n_periods]


def bootstrap_blended(windows, weights, n_periods, n_paths, block, rng, drift_haircut=0.0):
    """Single-series circular moving-block bootstrap with BLOCK-LEVEL era mixing.

    `windows` is a list of 1-D log-return arrays (one per era); `weights` are the
    probabilities of drawing each block from each era. A single simulated path can
    stitch a calm 2017 stretch onto a 2008-style crash block -- the future isn't
    locked into one regime. Returns (n_paths, n_periods+1) VALUE paths (incl. the
    day-0 start). A single window with weight 1.0 reduces to a plain moving-block
    bootstrap."""
    procs = []
    for w in windows:
        r = np.asarray(w, dtype=float)
        if drift_haircut:
            r = r - drift_haircut * r.mean()
        procs.append(r)
    pool = np.concatenate(procs)                                   # all eras end-to-end
    idx = _block_indices([len(r) for r in procs], weights, n_periods, n_paths, block, rng)
    cum = np.cumsum(pool[idx], axis=1)
    paths = AMOUNT * np.exp(cum)
    return np.concatenate([np.full((n_paths, 1), AMOUNT), paths], axis=1)


def bootstrap_terminal(log_returns, n_periods, n_paths, block, rng, drift_haircut=0.0):
    """Single-window moving-block bootstrap (thin wrapper over bootstrap_blended)."""
    return bootstrap_blended([log_returns], [1.0], n_periods, n_paths, block, rng, drift_haircut)


def bootstrap_portfolio(returns_matrix, weights_vec, n_periods, n_paths, block, rng, drift_haircut=0.0):
    """Correlation-preserving JOINT block bootstrap for a multi-asset portfolio.

    returns_matrix : (n_obs, n_assets) date-ALIGNED SIMPLE returns.
    weights_vec    : (n_assets,) portfolio weights (should sum to 1).

    One block-index array is drawn and applied to EVERY asset column, so each
    resampled timestep is a real historical cross-section -- the cross-asset
    correlation structure is preserved exactly, with no Cholesky / Gaussian
    assumption. Rebalanced EVERY period (daily for daily data) to constant weights:
    per-period portfolio simple return = cross-section @ weights. Returns
    (n_paths, n_periods+1) value paths. Reduces to the single-series engine when
    n_assets == 1 and weight == 1."""
    R = np.asarray(returns_matrix, dtype=float)
    if drift_haircut:
        R = R - drift_haircut * R.mean(axis=0)
    idx = _block_indices([R.shape[0]], [1.0], n_periods, n_paths, block, rng)
    port_r = R[idx] @ np.asarray(weights_vec, dtype=float)         # (n_paths, n_periods)
    paths = AMOUNT * np.cumprod(1.0 + port_r, axis=1)
    return np.concatenate([np.full((n_paths, 1), AMOUNT), paths], axis=1)


def analyze(value_paths, years):
    terminal = value_paths[:, -1]
    ret = terminal / AMOUNT - 1.0
    cagr = (terminal / AMOUNT) ** (1.0 / years) - 1.0

    runmax = np.maximum.accumulate(value_paths, axis=1)
    maxdd = (value_paths / runmax - 1.0).min(axis=1)

    tail_p = (1.0 - VAR_CONF) * 100.0          # e.g. 5
    var_ret = np.percentile(ret, tail_p)        # outcome at the tail percentile
    cvar_ret = ret[ret <= var_ret].mean()       # mean of the worst tail
    cash_factor = (1.0 + CASH_RATE) ** years

    return {
        "P(profit)":   float((terminal > AMOUNT).mean()),
        "P(beat cash)": float((terminal > AMOUNT * cash_factor).mean()),
        "val_P5":  np.percentile(terminal, 5),
        "val_P10": np.percentile(terminal, 10),
        "val_P50": np.percentile(terminal, 50),
        "val_P90": np.percentile(terminal, 90),
        "val_P95": np.percentile(terminal, 95),
        "cagr_P10": np.percentile(cagr, 10),
        "cagr_P50": np.percentile(cagr, 50),
        "cagr_P90": np.percentile(cagr, 90),
        "mean_cagr": float(cagr.mean()),
        "maxdd_med": float(np.percentile(maxdd, 50)),
        "maxdd_p95worst": float(np.percentile(maxdd, 5)),
        "var_ret": float(var_ret),
        "cvar_ret": float(cvar_ret),
    }


def build_fan(value_paths, ppy):
    """Downsample to ~monthly steps and return time-series percentile bands of
    portfolio value -- the probability cone for charting."""
    n = value_paths.shape[1]
    step = max(1, int(round(ppy / 12)))
    idx = np.array(sorted(set(list(range(0, n, step)) + [n - 1])))
    sub = value_paths[:, idx]
    qmap = {"p5": 5, "p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90, "p95": 95}
    fan = {"years": (idx / ppy).tolist()}
    for k, q in qmap.items():
        fan[k] = np.percentile(sub, q, axis=0).tolist()
    return fan


# ----------------------------------------------------------------------
# 5. ORCHESTRATION
# ----------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Block-bootstrap portfolio/index risk simulator with a neutral "
                    "P(profit)-vs-threshold indicator (not a trading signal).")
    p.add_argument("ticker", nargs="?", default=TICKER,
                   help="single ticker; a diversified fund (default: %(default)s) or a stock")
    p.add_argument("--portfolio", default=PORTFOLIO,
                   help='multi-asset allocation, e.g. "VTI:0.8,QQQ:0.2" (correlation-preserving)')
    p.add_argument("--csv", default=CSV_PATH, help="use a local CSV export instead of fetching")
    p.add_argument("--years", type=int, default=HISTORY_YEARS,
                   help="cap history to the last N years (single-window mode)")
    p.add_argument("--blend", action="store_true",
                   help="blended-window mode: mix return blocks across eras (see BLEND)")
    p.add_argument("--threshold", type=float, default=PROFIT_THRESHOLD,
                   help="P(profit) threshold the indicator flags above/below, 0-1 (default: %(default)s)")
    p.add_argument("--amount", type=float, default=AMOUNT, help="hypothetical invest amount")
    p.add_argument("--paths", type=int, default=N_PATHS, help="number of bootstrap paths")
    p.add_argument("--haircut", type=float, default=DRIFT_HAIRCUT,
                   help="fraction of historical drift to remove, 0-1 (stress test)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON (for an app)")
    return p.parse_args()


def compute(args):
    """Run the engine and return a structured results dict consumed by the text
    report and the --json output. Handles three input modes -- a diversified fund,
    a single stock (weaker prior), or a multi-asset --portfolio -- and returns the
    SAME output keys for all three so app.py / calibration.py stay agnostic."""
    global AMOUNT
    AMOUNT = args.amount                         # analyze() / bootstrap read this global
    rng = np.random.default_rng(SEED)
    portfolio = getattr(args, "portfolio", None)

    if portfolio:
        # -------- multi-asset: correlation-preserving joint bootstrap ----
        weights_map = parse_weights(portfolio)
        rdates, R, w, tickers, source = fetch_portfolio(weights_map, None)
        ppy, block = detect_frequency(rdates)
        port_simple = R @ w
        port_price = np.concatenate([[1.0], np.cumprod(1.0 + port_simple)])
        n_years_hist = (rdates[-1] - rdates[0]).days / 365.25
        full_drift, full_vol = annualize(np.log1p(port_simple), ppy)
        hist_mdd = historical_max_drawdown(port_price)
        worst_day = float(port_simple.min())
        obs, h_start, h_end = int(R.shape[0]), str(rdates[0]), str(rdates[-1])
        mode = "portfolio"
        comp = []
        for i, t in enumerate(tickers):
            d, v = annualize(np.log1p(R[:, i]), ppy)
            comp.append({"label": t, "years": None, "obs": obs, "weight": float(w[i]), "drift": d, "vol": v})
        warnings = [f"DIVERSIFIED PORTFOLIO ({len(tickers)} assets): bootstrap prior is statistically "
                    "appropriate; cross-asset correlation preserved by joint resampling."]
        if n_years_hist < 3:
            warnings.append(f"THIN OVERLAP (~{n_years_hist:.1f} yrs common history): ranges unreliable.")
        if hist_mdd > -0.25:
            warnings.append(f"NO REAL DRAWDOWN in overlap (worst {hist_mdd*100:.0f}%): likely all-bull; downside understated.")

        def simulate(n_periods):
            return bootstrap_portfolio(R, w, n_periods, args.paths, block, rng, args.haircut)
    else:
        # -------- single series (diversified fund OR single stock) -------
        dates, prices, source = load_prices(args.ticker, args.csv)
        log_ret = np.diff(np.log(prices))
        ppy, block = detect_frequency(dates)
        n_years_hist = (dates[-1] - dates[0]).days / 365.25
        full_drift, full_vol = annualize(log_ret, ppy)
        hist_mdd = historical_max_drawdown(prices)
        worst_day = float(np.expm1(log_ret.min()))
        obs, h_start, h_end = int(len(prices)), str(dates[0]), str(dates[-1])

        if args.blend:
            mode = "blended"
            wsum = sum(BLEND.values())
            comp, windows, weights = [], [], []
            for yrs, wt in BLEND.items():
                wr = slice_window(dates, log_ret, yrs)[1]
                d, v = annualize(wr, ppy)
                comp.append({"label": ("full" if yrs is None else f"{yrs}y"), "years": yrs,
                             "obs": int(len(wr)), "weight": wt / wsum, "drift": d, "vol": v})
                windows.append(wr); weights.append(wt / wsum)
            recent_years = min([y for y in BLEND if y is not None], default=None)
        else:
            mode = "full history" if args.years is None else f"last {args.years}y"
            wr = slice_window(dates, log_ret, args.years)[1]
            d, v = annualize(wr, ppy)
            windows, weights = [wr], [1.0]
            comp = [{"label": ("full" if args.years is None else f"{args.years}y"), "years": args.years,
                     "obs": int(len(wr)), "weight": 1.0, "drift": d, "vol": v}]
            recent_years = args.years

        warn_ret = slice_window(dates, log_ret, recent_years)[1] if recent_years is not None else log_ret
        warn_drift = annualize(warn_ret, ppy)[0]
        warn_years = recent_years if recent_years is not None else n_years_hist
        warnings = []
        if (not args.csv) and args.ticker.upper() not in DIVERSIFIED:
            warnings.append("WEAKER PRIOR: single stock -- a one-name bootstrap is idiosyncratic "
                            "(earnings/fraud/obsolescence it can't foresee). Prefer a diversified "
                            "index or --portfolio as the primary analysis.")
        elif args.csv:
            warnings.append("CSV input: if this is a single company (not a diversified fund), treat the prior as weak.")
        if warn_years < 3:
            warnings.append(f"THIN HISTORY (~{warn_years:.1f} yrs active): too few real regimes; ranges unreliable.")
        if hist_mdd > -0.25:
            warnings.append(f"NO REAL DRAWDOWN in history (worst {hist_mdd*100:.0f}%): likely all-bull; downside understated.")
        if warn_drift > 0.30:
            warnings.append(f"VERY HIGH drift ({warn_drift*100:.0f}%/yr) in the active window: extrapolates a boom; try --haircut.")
        if args.blend:
            warnings.append("BLEND active: recent-era weighting is balanced against older crash regimes (mitigates all-bull bias).")

        def simulate(n_periods):
            return bootstrap_blended(windows, weights, n_periods, args.paths, block, rng, args.haircut)

    # ---- simulate each horizon (mode-agnostic) -----------------------
    horizons = []
    fan = None
    max_h = max(HORIZONS_YEARS)
    for years in HORIZONS_YEARS:
        n_periods = int(round(years * ppy))
        vp = simulate(n_periods)
        r = analyze(vp, years)
        r["years"] = years
        r["n_periods"] = n_periods
        r["flag"] = "ABOVE" if r["P(profit)"] >= args.threshold else "BELOW"
        horizons.append(r)
        if years == max_h:
            fan = build_fan(vp, ppy)        # probability cone over the longest horizon

    return {
        "source": source, "mode": mode,
        "history": {"start": h_start, "end": h_end, "years": n_years_hist, "obs": obs,
                    "drift": full_drift, "vol": full_vol,
                    "max_drawdown": hist_mdd, "worst_day": worst_day},
        "windows": comp, "warnings": warnings,
        "params": {"amount": args.amount, "threshold": args.threshold, "paths": args.paths,
                   "block": block, "ppy": ppy, "haircut": args.haircut,
                   "cash_rate": CASH_RATE, "var_conf": VAR_CONF},
        "horizons": horizons, "fan": fan,
    }


def print_report(res):
    p = res["params"]; h = res["history"]; amt = p["amount"]; thr = p["threshold"]
    print("=" * 76)
    print("BLOCK-BOOTSTRAP PORTFOLIO RISK SIMULATOR")
    print("=" * 76)
    print(f"Data source      : {res['source']}")
    print(f"History          : {h['start']} -> {h['end']}  ({h['years']:.1f} yrs, {h['obs']} obs)")
    print(f"Annualized drift : {h['drift']*100:6.1f}%   (geometric, full history)")
    print(f"Annualized vol   : {h['vol']*100:6.1f}%")
    print(f"Worst single day : {h['worst_day']*100:6.1f}%")
    print(f"Max drawdown seen: {h['max_drawdown']*100:6.1f}%   (actual peak->trough in history)")
    print(f"Invest amount    : ${amt:,.0f}   |   Method: circular block bootstrap, "
          f"block={p['block']}, paths={p['paths']:,}")
    print(f"Sampling window  : {res['mode'].upper()}" + (f"   (drift haircut {p['haircut']*100:.0f}%)" if p['haircut'] else ""))
    if res["mode"] in ("blended", "portfolio") or len(res["windows"]) > 1 or res["windows"][0]["years"] is not None:
        label = "assets" if res["mode"] == "portfolio" else "window composition"
        print(f"  {label} feeding the simulation:")
        for w in res["windows"]:
            yl = w.get("label") or ("full" if w["years"] is None else f"{w['years']}y")
            print(f"     - {yl:<6} weight {w['weight']*100:3.0f}%  | {w['obs']:>5} obs | "
                  f"drift {w['drift']*100:+5.1f}% | vol {w['vol']*100:4.0f}%")

    if res["warnings"]:
        print("\n  !! HISTORY-QUALITY WARNINGS:")
        for w in res["warnings"]:
            print(f"     - {w}")

    print(f"\nIndicator        : flags whether P(profit) >= {thr*100:.0f}% (your threshold) -- "
          f"mechanical, NOT a buy/sell signal")
    print("All figures NOMINAL. Outputs are probabilistic ranges, never predictions.\n")

    for r in res["horizons"]:
        years = r["years"]
        print("#" * 76)
        print(f"#  HOLD {years} YEAR(S)        P(profit) {r['P(profit)']*100:.1f}%  "
              f"({r['flag']} the {thr*100:.0f}% threshold)")
        print("#" * 76)
        print(f"  Probability of profit          : {r['P(profit)']*100:5.1f}%")
        print(f"  Probability of beating cash@{p['cash_rate']*100:.0f}% : {r['P(beat cash)']*100:5.1f}%")
        print(f"  Value of ${amt:,.0f} after {years}y:")
        print(f"      pessimistic  P10 : ${r['val_P10']:>12,.0f}   ({(r['val_P10']/amt-1)*100:+.0f}%)")
        print(f"      median       P50 : ${r['val_P50']:>12,.0f}   ({(r['val_P50']/amt-1)*100:+.0f}%)")
        print(f"      optimistic   P90 : ${r['val_P90']:>12,.0f}   ({(r['val_P90']/amt-1)*100:+.0f}%)")
        print(f"  Annualized return (CAGR)       : "
              f"P10 {r['cagr_P10']*100:+5.1f}% | P50 {r['cagr_P50']*100:+5.1f}% | P90 {r['cagr_P90']*100:+5.1f}%")
        print(f"  Worst {(1-p['var_conf'])*100:.0f}% of outcomes (VaR)     : end <= {r['var_ret']*100:+.0f}%   "
              f"(CVaR, avg of that tail: {r['cvar_ret']*100:+.0f}%)")
        print(f"  Drawdown along the way         : "
              f"median {r['maxdd_med']*100:.0f}% | bad-case (worst 5%) {r['maxdd_p95worst']*100:.0f}%")
        print()

    print("=" * 76)
    print("READING THIS:")
    print("  P(profit)  = share of simulated futures ending above what you put in.")
    print("  P10/P90    = pessimistic / optimistic ends of the range (80% land between).")
    print("  VaR/CVaR   = the bad tail: where you end in the worst 5%, and its average.")
    print("  Drawdown   = worst peak-to-trough dip you'd sit through, even on good paths.")
    print("  The threshold flag is a mechanical comparison on a HISTORICAL prior, NOT")
    print("  advice and NOT a trading signal -- calibration shows P(profit) has no")
    print("  demonstrated skill vs a base rate. Use the distribution (VaR/CVaR/drawdown).")
    print("=" * 76)


def run():
    args = parse_args()
    res = compute(args)
    if args.json:
        print(json.dumps(res, indent=2, default=float))
    else:
        print_report(res)


if __name__ == "__main__":
    run()
