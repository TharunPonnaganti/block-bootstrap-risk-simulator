"""
Walk-forward calibration backtest for the block-bootstrap engine.

Checks whether the engine's probability forecasts match reality by stepping
through history: at each point, fit the model using ONLY past data, forecast
P(profit) for the next H years, then compare against what actually happened.

Scores with Brier Score, Brier Skill Score (vs out-of-sample base rate),
reliability curve, and PIT coverage. Reports both raw sample count and
effective independent N (overlapping windows are correlated).

Usage:
  python calibration.py SPY
  python calibration.py SPY --horizon 3
  python calibration.py --portfolio "VTI:0.8,QQQ:0.2"
"""
import json
import numpy as np
import stock_probability_engine as spe

PCT_LEVELS = [10, 50, 90]


# ----------------------------------------------------------------------
# Pure scoring functions (imported by qa_check.py -- test these directly)
# ----------------------------------------------------------------------
def brier_score(probs, outcomes):
    """Mean squared error of probability forecasts vs 0/1 outcomes. In [0, 1]."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def reliability_curve(probs, outcomes, n_bins=10):
    """Bucket forecasts into n_bins equal-width bins on [0,1]. Returns
    (predicted_mean, observed_freq, count) per bin; empty bins are NaN."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    pred = np.full(n_bins, np.nan)
    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            pred[b] = probs[m].mean()
            obs[b] = outcomes[m].mean()
    return pred, obs, cnt


def base_rate_brier(outcomes):
    """Brier score of the CLIMATOLOGY forecast (constant = unconditional base
    rate). Equals p(1-p) for base rate p -- the bar the model must beat."""
    outcomes = np.asarray(outcomes, dtype=float)
    base = float(outcomes.mean())
    return base * (1.0 - base), base


def brier_skill_score(probs, outcomes):
    """1 - Brier_model / Brier_baserate. >0 means the model beats climatology."""
    ref, _ = base_rate_brier(outcomes)
    if ref <= 0:
        return float("nan")
    return float(1.0 - brier_score(probs, outcomes) / ref)


def pit_coverage(forecast_pcts, realized, levels=PCT_LEVELS):
    """For each percentile level L, the fraction of realized outcomes at or below
    the forecast L-th percentile. Well-calibrated => coverage[L] ~= L/100."""
    realized = np.asarray(realized, dtype=float)
    return {L: float(np.mean(realized <= np.asarray(forecast_pcts[L]))) for L in levels}


# ----------------------------------------------------------------------
# Data loader -- single ticker, index, or portfolio (engine-agnostic)
# ----------------------------------------------------------------------
def load_series(ticker=None, portfolio=None, history_years=None):
    """Return (log_ret, ppy, block, source). For a portfolio, the rebalanced
    portfolio return series is built so calibration runs on one series."""
    if portfolio:
        weights = spe.parse_weights(portfolio)
        rdates, R, w, tickers, source, _native = spe.fetch_portfolio(weights, history_years)
        log_ret = np.log1p(R @ w)                       # rebalanced portfolio log-returns
        ppy, block = spe.detect_frequency(rdates)
        return log_ret, ppy, block, source
    dates, prices, source = spe.fetch_yahoo(ticker, history_years)
    log_ret = np.diff(np.log(prices))
    ppy, block = spe.detect_frequency(dates)
    return log_ret, ppy, block, source


# ----------------------------------------------------------------------
# Walk-forward driver
# ----------------------------------------------------------------------
def walk_forward(log_ret, ppy, block, h_years=1.0, step=63,
                 min_train_years=5.0, n_paths=2000, seed=42, min_base_windows=5):
    """Expanding-window walk-forward over `log_ret`. At each origin BOTH the model
    AND the benchmark use ONLY data available up to that origin (no look-ahead):
      - model     : block bootstrap fit on returns[:origin] -> P(profit).
      - benchmark : out-of-sample CLIMATOLOGY = frequency of positive H-period
                    returns observed strictly WITHIN returns[:origin]. (The earlier
                    version scored the model against the FULL-sample base rate, which
                    peeked at future data -- that look-ahead bias is removed here.)

    Origins whose training window holds fewer than `min_base_windows` NON-overlapping
    H-windows are EXCLUDED: with that little data the climatology rate is too unstable
    to benchmark against. Model and benchmark are scored on the same eligible set.

    NOTE: the pure helpers base_rate_brier / brier_skill_score implement the simpler
    CONSTANT full-sample climatology (used only by the unit tests in qa_check.py).
    The walk-forward BSS below uses the stricter per-origin OUT-OF-SAMPLE climatology."""
    H = int(round(h_years * ppy))
    min_train = int(round(min_train_years * ppy))
    n = len(log_ret)
    if n - H <= min_train:
        raise RuntimeError(f"Not enough history: need > {min_train + H} returns, have {n}.")

    rng = np.random.default_rng(seed)
    cal_amount = 1.0                                     # P(profit) is scale-free; use unit base
    C = np.concatenate([[0.0], np.cumsum(log_ret)])     # C[i] = sum(log_ret[:i]) for fast window sums
    base_eligible_oi = min_base_windows * H             # min training length for a stable base rate

    raw_origins = list(range(min_train, n - H, step))
    probs, bench, outcomes, realized_mult = [], [], [], []
    fpct = {L: [] for L in PCT_LEVELS}
    elig_origins = []
    n_excluded = 0
    for oi in raw_origins:
        if oi < base_eligible_oi:                       # too little prior data for a stable base rate
            n_excluded += 1
            continue
        train = log_ret[:oi]                            # only data available at the origin
        vp = spe.bootstrap_blended([train], [1.0], H, n_paths, block, rng, amount=cal_amount)
        mult = vp[:, -1] / cal_amount                    # terminal multiple distribution
        probs.append(float((mult > 1.0).mean()))
        # OUT-OF-SAMPLE benchmark: positive-H-window frequency in TRAINING data only.
        # window ending at j (covers returns[j-H:j]) is positive iff C[j] - C[j-H] > 0.
        win_pos = (C[H:oi + 1] - C[0:oi + 1 - H]) > 0
        bench.append(float(win_pos.mean()))
        for L in PCT_LEVELS:
            fpct[L].append(float(np.percentile(mult, L)))
        realized = float(np.exp(log_ret[oi:oi + H].sum()))   # actual price[t+H]/price[t]
        realized_mult.append(realized)
        outcomes.append(1.0 if realized > 1.0 else 0.0)
        elig_origins.append(oi)

    if not elig_origins:
        raise RuntimeError(
            f"No eligible origins: all {len(raw_origins)} raw origins were excluded because "
            f"their training windows had fewer than {min_base_windows} non-overlapping "
            f"{h_years}-year windows. Try a shorter horizon, more history, or fewer "
            f"min_base_windows.")

    probs = np.array(probs); bench = np.array(bench)
    outcomes = np.array(outcomes); realized_mult = np.array(realized_mult)
    fpct = {L: np.array(v) for L, v in fpct.items()}

    pred, obs, cnt = reliability_curve(probs, outcomes)
    bs_model = brier_score(probs, outcomes)
    bs_bench = brier_score(bench, outcomes)
    bss = float(1.0 - bs_model / bs_bench) if bs_bench > 0 else float("nan")
    cov = pit_coverage(fpct, realized_mult)

    span = elig_origins[-1] - elig_origins[0] + H
    n_eff = max(1, int(round(span / H)))                # effective non-overlapping samples (backtest)
    full_independent = int(n / H)                       # total history span / horizon

    return {
        "h_years": h_years, "H_periods": H, "ppy": ppy, "block": block,
        "n_origins_raw": len(raw_origins), "n_eligible": len(elig_origins),
        "n_excluded": n_excluded, "n_effective": n_eff, "full_independent": full_independent,
        "min_train_years": min_train_years, "step": step, "min_base_windows": min_base_windows,
        "bench_rate_mean": float(bench.mean()), "realized_rate": float(outcomes.mean()),
        "brier": bs_model, "brier_benchmark": bs_bench, "bss": bss, "coverage": cov,
        "reliability": {"predicted": pred.tolist(), "observed": obs.tolist(), "count": cnt.tolist()},
        "probs": probs, "outcomes": outcomes,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def plot_reliability(res, source, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.array(res["reliability"]["predicted"], dtype=float)
    obs = np.array(res["reliability"]["observed"], dtype=float)
    cnt = np.array(res["reliability"]["count"], dtype=float)
    m = ~np.isnan(pred)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="perfect calibration")
    sizes = 40 + 360 * (cnt[m] / cnt[m].max() if cnt[m].max() else 1)
    ax.scatter(pred[m], obs[m], s=sizes, color="#1b5e20", alpha=0.8, zorder=3,
               label="model (point size = # forecasts)")
    ax.axhline(res["bench_rate_mean"], color="#b71c1c", ls=":", lw=1.2,
               label=f"OOS base rate ~{res['bench_rate_mean']*100:.0f}%")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast P(profit)")
    ax.set_ylabel("Observed frequency of profit")
    ax.set_title(f"Reliability — {source}\n"
                 f"H={res['h_years']:g}y | Brier {res['brier']:.3f} | "
                 f"BSS {res['bss']:+.3f} | N={res['n_eligible']} (eff ~{res['n_effective']})",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def print_report(res, source):
    print("=" * 74)
    print("WALK-FORWARD CALIBRATION  (expanding window)")
    print("=" * 74)
    print(f"Series           : {source}")
    print(f"Horizon          : {res['h_years']:g} year(s)  ({res['H_periods']} periods)")
    print(f"Train min        : {res['min_train_years']:g}y expanding | origin step: {res['step']} periods")
    print(f"Origins          : {res['n_origins_raw']} raw | {res['n_eligible']} eligible "
          f"| {res['n_excluded']} excluded (training had < {res['min_base_windows']} non-overlapping windows)")
    print(f"INDEPENDENT N    : ~{res['n_effective']} backtested  |  ~{res['full_independent']} "
          f"in full history (span / horizon)")
    print("  ** All scores below are interpreted against the INDEPENDENT N, not the raw origin count.")
    print("     Overlapping windows are autocorrelated; single-digit independent N => wide error bars. **\n")

    print(f"Benchmark (OOS climatology, no look-ahead): mean rate {res['bench_rate_mean']*100:.1f}% "
          f"| realized rate {res['realized_rate']*100:.1f}%")
    print(f"Brier  model                  : {res['brier']:.4f}   (lower is better)")
    print(f"Brier  OOS benchmark          : {res['brier_benchmark']:.4f}")
    print(f"Brier Skill Score (vs OOS)    : {res['bss']:+.4f}")
    if res["bss"] > 0:
        print(f"  -> model edges the out-of-sample base rate, but with only ~{res['n_effective']} "
              f"independent windows this is NOT statistically conclusive.")
    else:
        print(f"  -> model does NOT beat the out-of-sample base rate. With only ~{res['n_effective']} "
              f"independent windows the estimate is noisy; read this as 'no demonstrated skill', "
              f"not 'proven negative skill'.")

    print("\nReliability (forecast bucket -> observed frequency):")
    print(f"  {'bucket':<12}{'n':>6}{'predicted':>12}{'observed':>12}")
    pred = res["reliability"]["predicted"]; obs = res["reliability"]["observed"]; cnt = res["reliability"]["count"]
    for b in range(len(cnt)):
        if cnt[b]:
            lo, hi = b * 10, (b + 1) * 10
            print(f"  {f'{lo}-{hi}%':<12}{cnt[b]:>6}{pred[b]*100:>11.1f}%{obs[b]*100:>11.1f}%")

    print("\nDistribution coverage (PIT) -- realized at/below forecast percentile:")
    for L, c in res["coverage"].items():
        print(f"  forecast P{L:<2}  covers {c*100:5.1f}%   (target {L}%)")
    print("=" * 74)


# ----------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Walk-forward calibration of the block-bootstrap engine.")
    p.add_argument("ticker", nargs="?", default="SPY", help="ticker (default: %(default)s)")
    p.add_argument("--portfolio", default=None, help='allocation, e.g. "VTI:0.8,QQQ:0.2"')
    p.add_argument("--horizon", type=float, default=1.0, help="forecast horizon in years")
    p.add_argument("--step", type=int, default=63, help="trading days between origins")
    p.add_argument("--min-train", type=float, default=5.0, help="minimum training window (years)")
    p.add_argument("--paths", type=int, default=2000, help="bootstrap paths per origin")
    p.add_argument("--out", default="reliability_curve.png", help="reliability PNG output path")
    p.add_argument("--json", action="store_true", help="also print results as JSON")
    return p.parse_args()


def run():
    args = parse_args()
    log_ret, ppy, block, source = load_series(args.ticker if not args.portfolio else None,
                                              args.portfolio)
    res = walk_forward(log_ret, ppy, block, h_years=args.horizon, step=args.step,
                       min_train_years=args.min_train, n_paths=args.paths)
    print_report(res, source)
    out = plot_reliability(res, source, args.out)
    print(f"\nReliability curve saved -> {out}")
    if args.json:
        slim = {k: v for k, v in res.items() if k not in ("probs", "outcomes")}
        print(json.dumps(slim, indent=2, default=float))


if __name__ == "__main__":
    run()
