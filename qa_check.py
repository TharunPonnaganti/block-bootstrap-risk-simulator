"""QA harness for stock_probability_engine. Checks correctness invariants
on real AAPL data plus edge/failure cases. Prints PASS/FAIL per check."""
import numpy as np
import stock_probability_engine as spe
import calibration as cal
import portfolio_construction as pc

PASS, FAIL = "PASS", "FAIL"
results = []
def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))

# ---- load real AAPL daily history once -------------------------------
dates, prices, source = spe.fetch_yahoo("AAPL", None)
log_ret = np.diff(np.log(prices))
ppy, block = spe.detect_frequency(dates)
spe.AMOUNT = 10_000.0
check("data: daily granularity (ppy==252, block==21)", ppy == 252.0 and block == 21,
      f"ppy={ppy}, block={block}, obs={len(prices)}")

n5 = int(round(5 * ppy))

# 1) DETERMINISM: same seed -> identical paths
a = spe.bootstrap_terminal(log_ret, n5, 4000, block, np.random.default_rng(7))
b = spe.bootstrap_terminal(log_ret, n5, 4000, block, np.random.default_rng(7))
check("determinism: identical output for identical seed", np.array_equal(a, b))

# 2) UNBIASED DRIFT: circular bootstrap samples uniformly -> E[sampled]==hist mean
rng = np.random.default_rng(0)
big = spe.bootstrap_terminal(log_ret, 5000, 4000, block, rng)  # value paths
sampled_logs = np.log(big[:, 1:] / big[:, :-1])
check("unbiased drift: bootstrap mean ~= historical mean return",
      abs(sampled_logs.mean() - log_ret.mean()) < 0.02 * abs(log_ret.mean()) + 1e-6,
      f"hist={log_ret.mean():.6e}  boot={sampled_logs.mean():.6e}")

# 3) DRAWDOWN sign: every path's max drawdown must be <= 0
vp = spe.bootstrap_terminal(log_ret, n5, 6000, block, np.random.default_rng(1))
runmax = np.maximum.accumulate(vp, axis=1)
maxdd = (vp / runmax - 1.0).min(axis=1)
check("drawdown: all max-drawdowns <= 0", bool((maxdd <= 1e-12).all()),
      f"max(maxdd)={maxdd.max():.3e}, min={maxdd.min():.3f}")

# 4) SCALE INVARIANCE of P(profit): independent of AMOUNT
spe.AMOUNT = 1_000.0
r_small = spe.analyze(spe.bootstrap_terminal(log_ret, n5, 8000, block, np.random.default_rng(3)), 5)
spe.AMOUNT = 5_000_000.0
r_big = spe.analyze(spe.bootstrap_terminal(log_ret, n5, 8000, block, np.random.default_rng(3)), 5)
spe.AMOUNT = 10_000.0
check("scale-invariance: P(profit) identical across invest amounts",
      abs(r_small["P(profit)"] - r_big["P(profit)"]) < 1e-12,
      f"$1k={r_small['P(profit)']:.4f}  $5M={r_big['P(profit)']:.4f}")

# 5) VaR/CVaR ordering: CVaR (avg of tail) must be <= VaR (tail edge)
r = spe.analyze(spe.bootstrap_terminal(log_ret, n5, 10000, block, np.random.default_rng(5)), 5)
check("tail risk: CVaR <= VaR", r["cvar_ret"] <= r["var_ret"] + 1e-12,
      f"VaR={r['var_ret']:.3f}  CVaR={r['cvar_ret']:.3f}")

# 6) INTERNAL CONSISTENCY: CAGR_P50 reconstructs from value_P50
implied = (r["val_P50"] / spe.AMOUNT) ** (1 / 5) - 1
check("consistency: CAGR_P50 == implied CAGR from val_P50",
      abs(implied - r["cagr_P50"]) < 1e-8,
      f"implied={implied:.10f} reported={r['cagr_P50']:.10f}")

# 7) MONOTONIC P(profit) with horizon for a positive-drift stock
ps = []
for yrs in (1, 3, 5):
    np_ = int(round(yrs * ppy))
    ps.append(spe.analyze(spe.bootstrap_terminal(log_ret, np_, 10000, block, np.random.default_rng(42)), yrs)["P(profit)"])
check("monotonic: P(profit) rises with horizon (AAPL)", ps[0] < ps[1] < ps[2], f"1/3/5y = {[round(p,3) for p in ps]}")

# 8) DRIFT_HAIRCUT=1.0 strips drift -> median CAGR ~0 and P(profit) drops toward ~0.5
r_h = spe.analyze(spe.bootstrap_terminal(log_ret, n5, 10000, block, np.random.default_rng(9), drift_haircut=1.0), 5)
check("stress: full drift haircut -> median CAGR near 0",
      abs(r_h["cagr_P50"]) < 0.03, f"median CAGR={r_h['cagr_P50']:.4f}")
check("stress: full drift haircut -> P(profit) drops below no-haircut",
      r_h["P(profit)"] < r["P(profit)"], f"haircut P(profit)={r_h['P(profit)']:.3f} vs base {r['P(profit)']:.3f}")

# 9) FAILURE MODE: invalid ticker raises cleanly
try:
    spe.fetch_yahoo("ZZ_NOT_A_TICKER_XX")
    check("error handling: invalid ticker raises", False, "no exception raised")
except Exception as e:
    check("error handling: invalid ticker raises", True, type(e).__name__)

# 10) ROBUSTNESS: history shorter than block length doesn't crash
tiny = log_ret[:10]
try:
    out = spe.bootstrap_terminal(tiny, 252, 100, block, np.random.default_rng(0))
    check("robustness: history < block length still runs", out.shape == (100, 253), f"shape={out.shape}")
except Exception as e:
    check("robustness: history < block length still runs", False, repr(e))

# ======================================================================
# PORTFOLIO (joint bootstrap) invariants
# ======================================================================
# 11) WEIGHTS normalize to 1 (explicit + equal-weight specs)
wmap = spe.parse_weights("VTI:3,QQQ:1")          # unnormalized on purpose
wv = np.array(list(wmap.values())); wv = wv / wv.sum()
wmap_eq = spe.parse_weights("A,B,C,D")
check("portfolio: weights normalize to 1",
      abs(wv.sum() - 1) < 1e-12 and abs(wv[0] - 0.75) < 1e-9
      and abs(sum(wmap_eq.values()) - 1) < 1e-12 and all(abs(v - 0.25) < 1e-9 for v in wmap_eq.values()))

# 11b) WEIGHT VALIDATION: reject zero, negative, mixed explicit/implicit
_bad_cases = [
    ("VTI:0,QQQ:1", "zero weight"),
    ("VTI:-0.5,QQQ:1.5", "negative weight"),
    ("VTI:0.8,QQQ", "mixed explicit/implicit"),
    ("", "empty spec"),
]
_all_rejected = True
for _spec, _reason in _bad_cases:
    try:
        spe.parse_weights(_spec)
        _all_rejected = False
    except (ValueError, Exception):
        pass
check("portfolio: weight validation rejects invalid specs",
      _all_rejected, f"tested: {[r for _, r in _bad_cases]}")

# 12) JOINT resampling PRESERVES cross-asset correlation; INDEPENDENT destroys it
_n, _rho = 4000, 0.6
_z0 = np.random.default_rng(1).standard_normal(_n)
_z1 = np.random.default_rng(11).standard_normal(_n)
_R = np.column_stack([0.0004 + 0.01 * _z0,
                      0.0004 + 0.01 * (_rho * _z0 + np.sqrt(1 - _rho**2) * _z1)])
_corr_hist = np.corrcoef(_R.T)[0, 1]
_idx = spe._block_indices([_n], [1.0], 2000, 500, block, np.random.default_rng(2))
_corr_joint = np.corrcoef(_R[_idx].reshape(-1, 2).T)[0, 1]
check("portfolio: joint resampling preserves correlation",
      abs(_corr_joint - _corr_hist) < 0.05, f"hist={_corr_hist:.3f} joint={_corr_joint:.3f}")
_ia = spe._block_indices([_n], [1.0], 2000, 500, block, np.random.default_rng(3))
_ib = spe._block_indices([_n], [1.0], 2000, 500, block, np.random.default_rng(4))
_corr_indep = np.corrcoef(np.column_stack([_R[:, 0][_ia].ravel(), _R[:, 1][_ib].ravel()]).T)[0, 1]
check("portfolio: independent resampling destroys correlation",
      abs(_corr_indep) < 0.10 and abs(_corr_indep) < abs(_corr_hist) / 2,
      f"hist={_corr_hist:.3f} independent={_corr_indep:.3f}")

# 13) SINGLE-ASSET reduction: bootstrap_portfolio (1 asset, w=1) == scalar engine
spe.AMOUNT = 10_000.0
_simple = np.expm1(log_ret)
_a = spe.bootstrap_terminal(log_ret, n5, 3000, block, np.random.default_rng(123))
_b = spe.bootstrap_portfolio(_simple[:, None], [1.0], n5, 3000, block, np.random.default_rng(123))
check("portfolio: single-asset reduces to the scalar engine",
      np.allclose(_a, _b, rtol=1e-9, atol=1e-6), f"max abs diff={np.abs(_a - _b).max():.2e}")

# ======================================================================
# PORTFOLIO CONSTRUCTION invariants (candidate allocations, not advice)
# ======================================================================
# 13b) Equal weight is exactly 1/n, long-only, sums to 1
_ew = pc.equal_weight(["VTI", "QQQ", "BND", "VXUS"])
check("construction: equal_weight returns 1/n long-only weights",
      np.allclose(_ew, 0.25) and abs(_ew.sum() - 1.0) < 1e-12 and (_ew >= 0).all(),
      f"weights={_ew}")

# 13c) Inverse-vol weights are normalized and give more weight to lower-vol assets
_base = np.linspace(-1.0, 1.0, 250)
_iv_R = np.column_stack([0.01 * _base, 0.02 * _base, 0.04 * _base])
_iv = pc.inverse_vol_weights(_iv_R)
_iv_expected = np.array([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0])
check("construction: inverse_vol_weights scale as 1/vol",
      np.allclose(_iv, _iv_expected, atol=1e-12)
      and abs(_iv.sum() - 1.0) < 1e-12 and (_iv >= 0).all(),
      f"weights={_iv}")

# 13d) Full ERC risk parity equalizes realized variance risk contributions
_rp_rng = np.random.default_rng(77)
_target_cov = np.array([[0.0400, 0.0140, 0.0020],
                        [0.0140, 0.0225, 0.0060],
                        [0.0020, 0.0060, 0.0100]])
_rp_R = _rp_rng.multivariate_normal(np.zeros(3), _target_cov, size=5000)
_rp_w = pc.risk_parity_weights(_rp_R)
_rp_cov = pc.covariance_matrix(_rp_R)
_rp_rc = pc.risk_contributions(_rp_w, _rp_cov)
check("construction: risk_parity_weights equalize risk contributions",
      abs(_rp_w.sum() - 1.0) < 1e-12 and (_rp_w >= 0).all()
      and np.max(np.abs(_rp_rc - 1.0 / 3.0)) < 1e-6,
      f"weights={np.round(_rp_w, 4)} risk_contrib={np.round(_rp_rc, 6)}")

# 13e) Concentration metrics: HHI and effective number of bets
check("construction: HHI/effective bets are consistent",
      abs(pc.hhi(_ew) - 0.25) < 1e-12
      and abs(pc.effective_number_of_bets(_ew) - 4.0) < 1e-12)

# 13f) Min-CVaR reduces empirical in-sample CVaR loss vs equal weight on a fat-tail toy set
_tail_rng = np.random.default_rng(12)
_safe = _tail_rng.normal(0.0002, 0.002, 800)
_risky = _tail_rng.normal(0.0008, 0.012, 800)
_risky[:40] -= 0.08
_tail_R = np.column_stack([_risky, _safe])
_tail_eq = pc.equal_weight(2)
_tail_mc, _tail_info = pc.min_cvar_weights(_tail_R, solver="subgradient", return_info=True)
check("construction: min_cvar_weights lowers empirical CVaR loss vs equal weight",
      abs(_tail_mc.sum() - 1.0) < 1e-12 and (_tail_mc >= -1e-12).all()
      and pc.empirical_cvar_loss(_tail_R, _tail_mc) < pc.empirical_cvar_loss(_tail_R, _tail_eq),
      f"solver={_tail_info['solver']} ew={pc.empirical_cvar_loss(_tail_R, _tail_eq):.4f} "
      f"min={pc.empirical_cvar_loss(_tail_R, _tail_mc):.4f}")

# 13g) Mean-variance baseline returns valid long-only simplex weights
_mv_w = pc.mean_variance_weights(_rp_R)
check("construction: mean_variance_weights returns valid long-only weights",
      abs(_mv_w.sum() - 1.0) < 1e-12 and (_mv_w >= -1e-12).all(),
      f"weights={np.round(_mv_w, 4)}")

# 13h) Resampled efficient frontier returns a weight-dispersion cloud
_front_R = _rp_rng.multivariate_normal(
    np.array([0.0003, 0.0005, 0.0007]),
    np.array([[0.0300, 0.0100, 0.0040],
              [0.0100, 0.0250, 0.0060],
              [0.0040, 0.0060, 0.0200]]),
    size=250)
_front = pc.resampled_efficient_frontier(
    _front_R, risk_aversions=(3.0, 8.0), n_resamples=4, block=21, seed=13)
check("construction: resampled efficient frontier exposes MV weight dispersion",
      len(_front) == 8
      and all(abs(sum(r["weights"]) - 1.0) < 1e-12 for r in _front)
      and all("ann_return" in r and "ann_vol" in r and "weight_std_mean" in r for r in _front)
      and max(r["weight_std_mean"] for r in _front) >= 0.0)

# 13i) Candidate evaluation labels in-sample rows explicitly and returns lab metrics
_eval_rows = pc.evaluate_candidates(
    _rp_R[:3000], candidates=["equal_weight", "inverse_vol", "risk_parity", "min_cvar", "mean_variance"],
    years=1, n_paths=300, block=21, stability_resamples=5, seed=5)
_required = {"candidate", "label", "evaluation", "weights", "P(profit)", "P(beat cash)",
             "val_P10", "val_P50", "var_ret", "cvar_ret", "maxdd_med", "maxdd_p95worst",
             "hhi", "effective_bets", "weight_std_mean", "weight_std_max", "n_resamples"}
check("construction: evaluate_candidates marks in-sample and returns comparison metrics",
      len(_eval_rows) == 5
      and all(r["evaluation"] == "in-sample" and r["label"].endswith("(in-sample)") for r in _eval_rows)
      and all(_required.issubset(r.keys()) for r in _eval_rows)
      and all(abs(sum(r["weights"]) - 1.0) < 1e-12 for r in _eval_rows)
      and all(r["effective_bets"] >= 1.0 for r in _eval_rows),
      f"rows={[r['label'] for r in _eval_rows]}")

# 13j) Supplying eval_returns flips the label without changing fit/eval asset contract
_oos_rows = pc.evaluate_candidates(
    _rp_R[:2500], eval_returns=_rp_R[2500:3500], candidates=["equal_weight"],
    years=1, n_paths=200, block=21, stability_resamples=3, seed=6)
check("construction: evaluate_candidates supports out-of-sample eval_returns hook",
      len(_oos_rows) == 1
      and _oos_rows[0]["evaluation"] == "out-of-sample"
      and not _oos_rows[0]["label"].endswith("(in-sample)")
      and _oos_rows[0]["n_resamples"] == 3)

# 13k) Chronological train/eval demo exercises the OOS path directly
_split_demo = pc.evaluate_train_eval_split(
    _rp_R[:1200], train_frac=0.65, candidates=["equal_weight", "risk_parity"],
    years=1, n_paths=150, block=21, stability_resamples=2, seed=7)
check("construction: train/eval split demo produces out-of-sample rows",
      _split_demo["train_obs"] > _split_demo["eval_obs"]
      and len(_split_demo["rows"]) == 2
      and all(r["evaluation"] == "out-of-sample" for r in _split_demo["rows"]))

# 13l) Candidate evaluation uses the same bootstrap/analyze engine as the core simulator
_eq_w = pc.equal_weight(3)
_direct_vp = spe.bootstrap_portfolio(
    _rp_R[:1000], _eq_w, int(round(1 * 252)), 200, 21, np.random.default_rng(88), amount=12_345.0)
_direct_metrics = spe.analyze(_direct_vp, 1, amount=12_345.0, cash_rate=0.03, var_conf=0.90)
_lab_row = pc.evaluate_candidates(
    _rp_R[:1000], candidates=["equal_weight"], years=1, n_paths=200, block=21,
    amount=12_345.0, cash_rate=0.03, var_conf=0.90, stability_resamples=0, seed=88)[0]
_engine_keys = ["P(profit)", "P(beat cash)", "val_P10", "val_P50",
                "var_ret", "cvar_ret", "maxdd_med", "maxdd_p95worst"]
check("construction: evaluate_candidates reuses core bootstrap/analyze engine",
      all(abs(_lab_row[k] - _direct_metrics[k]) < 1e-12 for k in _engine_keys),
      f"checked={_engine_keys}")

# ======================================================================
# CALIBRATION scoring invariants (synthetic, deterministic)
# ======================================================================
_rng = np.random.default_rng(0)
# 14) Brier: a perfect forecast scores 0
_o = np.array([0, 1, 0, 1, 1, 0], dtype=float)
check("calibration: perfect forecast -> Brier 0", cal.brier_score(_o, _o) == 0.0)

# 15) Brier: constant 0.5 on fair coin flips -> ~0.25
_cf = (_rng.random(40000) < 0.5).astype(float)
check("calibration: 0.5 on coin flips -> Brier ~0.25",
      abs(cal.brier_score(np.full(_cf.size, 0.5), _cf) - 0.25) < 0.01)

# 16) Base-rate Brier == p(1-p)
_o7 = (_rng.random(60000) < 0.7).astype(float)
_ref, _base = cal.base_rate_brier(_o7)
check("calibration: base-rate Brier == p(1-p)",
      abs(_ref - _base * (1 - _base)) < 1e-12 and abs(_base - 0.7) < 0.01)

# 17) BSS == 0 when the model IS the base-rate benchmark
check("calibration: BSS=0 when model==base rate",
      abs(cal.brier_skill_score(np.full(_o7.size, _o7.mean()), _o7)) < 1e-9)

# 18) BSS > 0 for a genuinely informative forecaster
_sig = _rng.random(40000)
_tp = 0.2 + 0.6 * _sig                     # true conditional probability
_out = (_rng.random(40000) < _tp).astype(float)
check("calibration: informative forecaster beats base rate (BSS>0)",
      cal.brier_skill_score(_tp, _out) > 0.05, f"BSS={cal.brier_skill_score(_tp, _out):.3f}")

# 19) Reliability bucket counts sum to N
_pred, _obs, _cnt = cal.reliability_curve(_tp, _out)
check("calibration: reliability bucket counts sum to N", int(_cnt.sum()) == _out.size)

# ======================================================================
# REGRESSION TESTS for the diagnostic-run fixes (classification + warnings)
# All inputs synthetic -> deterministic, no network.
# ======================================================================
# 20) (#1) Diversified classification: broad ETFs that were previously MISSING are
#     now classified diversified; genuine single names are not.
_should_be_div = ["QQQM", "VOO", "IVV", "VTI", "SPLG", "IJH", "IJR", "VT", "QQQ", "SCHD"]
_should_be_single = ["AAPL", "NVDA", "TSLA", "RELIANCE.NS"]
check("fix#1: broad ETFs (incl. QQQM) classify as diversified",
      all(spe.is_diversified(t) for t in _should_be_div),
      f"misclassified: {[t for t in _should_be_div if not spe.is_diversified(t)]}")
check("fix#1: single names classify as single (weak prior)",
      all(not spe.is_diversified(t) for t in _should_be_single),
      f"misclassified: {[t for t in _should_be_single if spe.is_diversified(t)]}")
check("fix#1: --prior override forces classification both ways",
      spe.is_diversified("NVDA", "diversified") is True
      and spe.is_diversified("VOO", "single") is False)

# 21) (#2) Thin-history fires when total history < 2x max horizon (e.g. 5.7y vs 5y),
#     and stays silent for long histories. max horizon here = max(HORIZONS_YEARS).
_mh = max(spe.HORIZONS_YEARS)
check("fix#2: thin-history warning fires on short history (5.7y vs 5y horizon)",
      spe.thin_history_warning(5.7, _mh) is not None
      and "THIN HISTORY" in spe.thin_history_warning(5.7, _mh))
check("fix#2: thin-history warning silent on long history (25y)",
      spe.thin_history_warning(25.0, _mh) is None)

# 22) (#3+#4) Thin-overlap fires when common overlap << longest single-asset history,
#     naming the limiting asset; silent when overlaps are comparable.
_nat = {"NVDA": 27.4, "VTI": 25.0, "QQQM": 5.7}
_ov = spe.thin_overlap_warning(5.7, _nat)
check("fix#3: thin-overlap fires on truncated portfolio and names limiter",
      _ov is not None and "THIN OVERLAP" in _ov and "QQQM" in _ov and "EXCLUDED" in _ov,
      _ov or "no warning")
check("fix#3: thin-overlap silent when histories are comparable",
      spe.thin_overlap_warning(24.0, {"VTI": 25.0, "VOO": 24.5}) is None)

# 23) (degenerate blend) fires when history < largest finite BLEND window; silent otherwise.
_largest_blend = max(y for y in spe.BLEND if y is not None)
check("fix: degenerate-blend warning fires on short history",
      spe.degenerate_blend_warning(5.7) is not None
      and "DEGENERATE" in spe.degenerate_blend_warning(5.7),
      f"largest finite blend window = {_largest_blend}y")
check("fix: degenerate-blend warning silent on long history",
      spe.degenerate_blend_warning(25.0) is None)

spe.AMOUNT = 10_000.0   # restore

# ---- report ----------------------------------------------------------
print("=" * 70)
print("QA RESULTS  (engine + portfolio + calibration invariants)")
print("=" * 70)
for status, name, detail in results:
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
n_fail = sum(1 for s, _, _ in results if s == FAIL)
print("-" * 70)
print(f"  {len(results)-n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f"   ({n_fail} FAILED)"))
print("=" * 70)
