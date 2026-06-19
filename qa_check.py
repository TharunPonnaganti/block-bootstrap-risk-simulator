"""QA harness for stock_probability_engine. Checks correctness invariants
on real AAPL data plus edge/failure cases. Prints PASS/FAIL per check."""
import numpy as np
import stock_probability_engine as spe
import calibration as cal

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
      abs(implied - r["cagr_P50"]) < 1e-9, f"implied={implied:.4f} reported={r['cagr_P50']:.4f}")

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
