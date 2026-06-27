"""Strict mathematical QA for the portfolio-construction lab (portfolio_construction.py).

Unlike qa_check.py (which validates the engine on real market data), this suite is
100% deterministic and checks CLOSED-FORM / statistical ground truth on synthetic
data with KNOWN structure:

  - exact-covariance returns are constructed so sample covariance == a target matrix
    (orthonormal QR basis scaled by Cholesky), making ERC/risk-parity identities EXACT
    rather than approximate;
  - every optimizer is checked against an analytical property it must satisfy
    (equal risk contributions, min-variance ordering, CVaR optimality bound, simplex
    projection optimality), not just "runs without error".

SciPy is optional: the exact Rockafellar-Uryasev LP tests run only if SciPy is present
(otherwise they are SKIPPED, not failed). The pure-NumPy subgradient path is always tested.

Run:  python qa_construction.py    (exit code 0 iff all non-skipped checks pass)
"""
import numpy as np
import portfolio_construction as pc
import stock_probability_engine as spe

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []
def check(name, cond, detail=""):
    results.append((PASS if cond else FAIL, name, detail))
def skip(name, detail=""):
    results.append((SKIP, name, detail))


# ----------------------------------------------------------------------
# Synthetic data with EXACT, known sample structure
# ----------------------------------------------------------------------
def exact_cov_returns(cov, n_obs=600, seed=0, mean=None):
    """Return an (n_obs x k) matrix whose SAMPLE covariance (ddof=1) equals `cov`
    EXACTLY (to machine precision). Built from an orthonormal QR basis of a centered
    Gaussian block, rescaled by the Cholesky factor: R = Q @ (sqrt(n-1) * L^T),
    where cov = L L^T and Q has orthonormal, centered columns."""
    cov = np.asarray(cov, dtype=float)
    k = cov.shape[0]
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_obs, k))
    M = M - M.mean(axis=0)
    Q, _ = np.linalg.qr(M)                 # orthonormal columns, centered
    L = np.linalg.cholesky(cov)            # cov = L L^T
    R = Q[:, :k] @ (np.sqrt(n_obs - 1) * L.T)
    if mean is not None:
        R = R + np.asarray(mean, dtype=float)
    return R

def cov_from(sigmas, corr):
    s = np.asarray(sigmas, dtype=float)
    C = np.asarray(corr, dtype=float)
    return np.outer(s, s) * C


# ======================================================================
# SECTION 1 -- exact-cov construction is itself correct (foundation)
# ======================================================================
_C = cov_from([0.01, 0.02, 0.04], [[1, 0.3, -0.2], [0.3, 1, 0.5], [-0.2, 0.5, 1]])
_R = exact_cov_returns(_C, n_obs=800, seed=1)
_S = pc.covariance_matrix(_R)
check("foundation: constructed sample covariance == target (machine precision)",
      np.max(np.abs(_S - _C)) < 1e-12, f"max|S-C|={np.max(np.abs(_S - _C)):.2e}")

# ======================================================================
# SECTION 2 -- weight validity for EVERY candidate (long-only simplex)
# ======================================================================
for _name, _w in pc.candidate_weights(_R).items():
    check(f"validity: {_name} sums to 1, long-only, correct length",
          abs(_w.sum() - 1.0) < 1e-8 and (_w >= -1e-9).all() and _w.size == 3,
          f"sum={_w.sum():.12f} min={_w.min():.2e}")

# ======================================================================
# SECTION 3 -- equal weight & concentration metrics (closed form)
# ======================================================================
for _n in (2, 3, 7):
    _ew = pc.equal_weight(_n)
    check(f"equal_weight: exactly 1/n (n={_n})", np.allclose(_ew, 1.0 / _n, atol=1e-15))
    check(f"HHI(equal)==1/n (n={_n})", abs(pc.hhi(_ew) - 1.0 / _n) < 1e-15)
    check(f"effective_bets(equal)==n (n={_n})", abs(pc.effective_number_of_bets(_ew) - _n) < 1e-12)
_conc = np.array([1.0, 0.0, 0.0, 0.0])
check("HHI(concentrated)==1 and effective_bets==1",
      abs(pc.hhi(_conc) - 1.0) < 1e-15 and abs(pc.effective_number_of_bets(_conc) - 1.0) < 1e-12)
# HHI bounds 1/n <= HHI <= 1 on random simplex weights
_rngb = np.random.default_rng(7)
_bound_ok = True
for _ in range(200):
    _wv = _rngb.dirichlet(np.ones(5))
    if not (1.0 / 5 - 1e-12 <= pc.hhi(_wv) <= 1.0 + 1e-12):
        _bound_ok = False
check("HHI bounds: 1/n <= HHI <= 1 (200 random simplex draws)", _bound_ok)

# ======================================================================
# SECTION 4 -- inverse volatility (closed form)
# ======================================================================
_sig = np.array([0.005, 0.01, 0.02, 0.05])
_Riv = exact_cov_returns(np.diag(_sig ** 2), n_obs=600, seed=2)
_iv = pc.inverse_vol_weights(_Riv)
_prod = _iv * _Riv.std(axis=0, ddof=1)
check("inverse_vol: w_i * sigma_i is constant across assets",
      np.allclose(_prod, _prod[0], rtol=1e-10), f"cv={_prod.std() / _prod.mean():.2e}")
check("inverse_vol: higher vol -> strictly lower weight (monotone)",
      np.all(np.diff(_iv) < 0), f"w={_iv.round(4)}")

# ======================================================================
# SECTION 5 -- risk parity / ERC (analytical ground truth)
# ======================================================================
# 5a) defining property: equal risk contributions, exact covariance, correlated
_Cerc = cov_from([0.01, 0.02, 0.03, 0.05],
                 [[1, .4, .2, .1], [.4, 1, .3, .2], [.2, .3, 1, .5], [.1, .2, .5, 1]])
_Rerc = exact_cov_returns(_Cerc, n_obs=900, seed=3)
_rp = pc.risk_parity_weights(_Rerc)
_rc = pc.risk_contributions(_rp, pc.covariance_matrix(_Rerc))
check("ERC: all risk contributions equal 1/n (exact correlated cov)",
      np.allclose(_rc, 1.0 / 4, atol=1e-8), f"rc={_rc.round(8)}")

# 5b) n=2 closed form: ERC weights ∝ 1/sigma for ANY correlation
_two_ok = True
for _rho in (-0.6, -0.2, 0.0, 0.35, 0.85):
    _C2 = cov_from([0.01, 0.03], [[1, _rho], [_rho, 1]])
    _R2 = exact_cov_returns(_C2, n_obs=500, seed=11)
    _rp2 = pc.risk_parity_weights(_R2)
    _iv2 = pc.inverse_vol_weights(_R2)
    if not np.allclose(_rp2, _iv2, atol=1e-7):
        _two_ok = False
check("ERC: n=2 equals inverse-vol for every correlation (closed form)", _two_ok)

# 5c) diagonal cov, n>2: ERC reduces to inverse-vol exactly
_Rdiag = exact_cov_returns(np.diag([0.005, 0.01, 0.02, 0.04]) ** 2, n_obs=700, seed=4)
check("ERC: diagonal covariance reduces to inverse-vol (n=4)",
      np.allclose(pc.risk_parity_weights(_Rdiag), pc.inverse_vol_weights(_Rdiag), atol=1e-7))

# 5d) determinism
check("ERC: deterministic for identical input",
      np.array_equal(pc.risk_parity_weights(_Rerc), pc.risk_parity_weights(_Rerc)))

# ======================================================================
# SECTION 6 -- mean-variance (optimality direction)
# ======================================================================
_Cmv = cov_from([0.01, 0.015, 0.02, 0.04],
                [[1, .2, .1, 0], [.2, 1, .3, .1], [.1, .3, 1, .2], [0, .1, .2, 1]])
_Rmv = exact_cov_returns(_Cmv, n_obs=900, seed=5, mean=[0.0003, 0.0004, 0.0005, 0.0006])
_S_mv = pc.covariance_matrix(_Rmv)
def _pvar(w): return float(w @ _S_mv @ w)
_w_hi = pc.mean_variance_weights(_Rmv, risk_aversion=1000.0)
_w_mid = pc.mean_variance_weights(_Rmv, risk_aversion=20.0)
_w_lo = pc.mean_variance_weights(_Rmv, risk_aversion=2.0)
check("MV: high risk-aversion variance <= equal-weight variance (min-var direction)",
      _pvar(_w_hi) <= _pvar(pc.equal_weight(4)) + 1e-14,
      f"mv_hi={_pvar(_w_hi):.3e} ew={_pvar(pc.equal_weight(4)):.3e}")
check("MV: portfolio variance monotonically decreases as risk-aversion rises",
      _pvar(_w_hi) <= _pvar(_w_mid) + 1e-12 <= _pvar(_w_lo) + 2e-12,
      f"var(2,20,1000)={_pvar(_w_lo):.3e},{_pvar(_w_mid):.3e},{_pvar(_w_hi):.3e}")
check("MV: deterministic for identical input",
      np.array_equal(pc.mean_variance_weights(_Rmv), pc.mean_variance_weights(_Rmv)))

# ======================================================================
# SECTION 7 -- simplex projection (Duchi et al.)
# ======================================================================
_rngp = np.random.default_rng(9)
_v = _rngp.standard_normal(6) * 3.0
_p = pc._project_simplex(_v)
check("simplex: projection sums to 1 and is non-negative",
      abs(_p.sum() - 1.0) < 1e-12 and (_p >= -1e-15).all())
check("simplex: projection is idempotent", np.allclose(pc._project_simplex(_p), _p, atol=1e-12))
_son = _rngp.dirichlet(np.ones(6))
check("simplex: a point already on the simplex is unchanged",
      np.allclose(pc._project_simplex(_son), _son, atol=1e-10))
# Euclidean optimality (necessary): projection is closer than random simplex points
_opt_ok = True
_dist_proj = np.linalg.norm(_p - _v)
for _ in range(500):
    _s = _rngp.dirichlet(np.ones(6))
    if np.linalg.norm(_s - _v) < _dist_proj - 1e-12:
        _opt_ok = False
check("simplex: projection is the nearest simplex point (vs 500 random points)", _opt_ok)

# ======================================================================
# SECTION 8 -- empirical CVaR + Min-CVaR optimality
# ======================================================================
_Rcv = exact_cov_returns(_Cerc, n_obs=1500, seed=6, mean=[0.0003, 0.0002, 0.0004, 0.0001])
_ewcv = pc.equal_weight(4)
_losses = -(_Rcv @ _ewcv)
_var = float(np.quantile(_losses, 0.95))
_cvar = pc.empirical_cvar_loss(_Rcv, _ewcv, 0.95)
check("CVaR: tail mean >= VaR (CVaR >= VaR)", _cvar >= _var - 1e-12, f"VaR={_var:.5f} CVaR={_cvar:.5f}")
check("CVaR: equals mean of losses at-or-beyond VaR (identity)",
      abs(_cvar - _losses[_losses >= _var].mean()) < 1e-12)
check("CVaR: monotone in alpha (deeper tail -> larger CVaR)",
      pc.empirical_cvar_loss(_Rcv, _ewcv, 0.90)
      <= pc.empirical_cvar_loss(_Rcv, _ewcv, 0.95) + 1e-12
      <= pc.empirical_cvar_loss(_Rcv, _ewcv, 0.99) + 1e-12)

# Min-CVaR optimality bound: subgradient solution never worse than equal-weight start
_w_sg = pc.min_cvar_weights(_Rcv, solver="subgradient")
_cv_sg = pc.empirical_cvar_loss(_Rcv, _w_sg, 0.95)
_cv_ew = pc.empirical_cvar_loss(_Rcv, _ewcv, 0.95)
check("Min-CVaR (subgradient): CVaR <= equal-weight CVaR (optimality bound, always holds)",
      _cv_sg <= _cv_ew + 1e-9, f"sg={_cv_sg:.6f} ew={_cv_ew:.6f}")
check("Min-CVaR (subgradient): weights valid simplex",
      abs(_w_sg.sum() - 1) < 1e-6 and (_w_sg >= -1e-9).all())
check("Min-CVaR (subgradient): deterministic for identical input/seed",
      np.array_equal(pc.min_cvar_weights(_Rcv, solver="subgradient"),
                     pc.min_cvar_weights(_Rcv, solver="subgradient")))

# Exact Rockafellar-Uryasev LP (only if SciPy installed)
if HAVE_SCIPY:
    _w_lp, _info = pc.min_cvar_weights(_Rcv, solver="linprog", return_info=True)
    _cv_lp = pc.empirical_cvar_loss(_Rcv, _w_lp, 0.95)
    check("Min-CVaR (LP): weights valid simplex",
          abs(_w_lp.sum() - 1) < 1e-6 and (_w_lp >= -1e-9).all())
    check("Min-CVaR (LP): exact optimum <= equal-weight CVaR",
          _cv_lp <= _cv_ew + 1e-9, f"lp={_cv_lp:.6f} ew={_cv_ew:.6f}")
    check("Min-CVaR (LP): exact optimum <= subgradient heuristic CVaR",
          _cv_lp <= _cv_sg + 1e-6, f"lp={_cv_lp:.6f} sg={_cv_sg:.6f}")
    check("Min-CVaR: subgradient within 5% of the exact LP optimum",
          _cv_sg <= _cv_lp * 1.05 + 1e-9, f"gap={(_cv_sg/_cv_lp - 1)*100:.2f}%")
    check("Min-CVaR (LP): solver tag reported as 'linprog'", _info["solver"] == "linprog")
else:
    skip("Min-CVaR (LP): exact Rockafellar-Uryasev path", "SciPy not installed (optional dep)")

# ======================================================================
# SECTION 9 -- resampled efficient frontier (fragility diagnostic)
# ======================================================================
_ra = (2.0, 5.0, 10.0, 50.0)
_front = pc.resampled_efficient_frontier(_Rmv, risk_aversions=_ra, n_resamples=30, seed=8)
check("frontier: row count == len(risk_aversions) * n_resamples",
      len(_front) == len(_ra) * 30, f"rows={len(_front)}")
check("frontier: every row carries return/vol/weights and valid simplex weights",
      all("ann_return" in r and "ann_vol" in r
          and abs(sum(r["weights"]) - 1) < 1e-6 and min(r["weights"]) >= -1e-9
          for r in _front))
check("frontier: weights disperse across resamples (fragility is demonstrated, std>0)",
      max(r["weight_std_mean"] for r in _front) > 1e-6,
      f"max weight_std_mean={max(r['weight_std_mean'] for r in _front):.2e}")
_meanvol = {ra: np.mean([r["ann_vol"] for r in _front if r["risk_aversion"] == ra]) for ra in _ra}
check("frontier: mean annualized vol falls as risk-aversion rises (ordering)",
      _meanvol[2.0] >= _meanvol[50.0] - 1e-12,
      f"vol(ra=2)={_meanvol[2.0]:.4f} vol(ra=50)={_meanvol[50.0]:.4f}")

# ======================================================================
# SECTION 10 -- evaluate_candidates: ENGINE EQUIVALENCE (no parallel engine)
# ======================================================================
_Req = exact_cov_returns(_Cerc, n_obs=800, seed=12, mean=[0.0004, 0.0003, 0.0005, 0.0002])
_seed, _paths, _yrs, _amt, _cr, _vc = 123, 1500, 5, 10_000.0, 0.04, 0.95
_rows = pc.evaluate_candidates(_Req, candidates=["equal_weight"], years=_yrs, n_paths=_paths,
                               amount=_amt, cash_rate=_cr, var_conf=_vc, seed=_seed,
                               stability_resamples=0)
_row = _rows[0]
# reproduce the SAME numbers directly through the engine with a fresh rng at the same seed
_rng = np.random.default_rng(_seed)
_w_ew = pc.equal_weight(4)
_np = int(round(_yrs * 252))
_vp = spe.bootstrap_portfolio(_Req, _w_ew, _np, _paths, 21, _rng, amount=_amt)
_m = spe.analyze(_vp, _yrs, amount=_amt, cash_rate=_cr, var_conf=_vc)
check("evaluate_candidates: equals direct engine bootstrap+analyze (one engine, byte-for-byte)",
      abs(_row["P(profit)"] - _m["P(profit)"]) < 1e-12
      and abs(_row["val_P50"] - _m["val_P50"]) < 1e-6
      and abs(_row["var_ret"] - _m["var_ret"]) < 1e-12
      and abs(_row["cvar_ret"] - _m["cvar_ret"]) < 1e-12,
      f"dP={abs(_row['P(profit)'] - _m['P(profit)']):.2e}")

# in-sample vs out-of-sample labelling
_in = pc.evaluate_candidates(_Req, candidates=["risk_parity"], n_paths=500, stability_resamples=0)
_out = pc.evaluate_candidates(_Req, eval_returns=_Req, candidates=["risk_parity"],
                              n_paths=500, stability_resamples=0)
check("evaluate_candidates: in-sample row is labelled '(in-sample)'",
      _in[0]["evaluation"] == "in-sample" and _in[0]["label"].endswith("(in-sample)"))
check("evaluate_candidates: passing eval_returns flips label to out-of-sample",
      _out[0]["evaluation"] == "out-of-sample" and "(in-sample)" not in _out[0]["label"])
check("evaluate_candidates: rows expose hhi, effective_bets, and tail metrics",
      all(k in _in[0] for k in ("hhi", "effective_bets", "P(profit)", "var_ret", "cvar_ret", "maxdd_p95worst")))

# ======================================================================
# SECTION 11 -- train/eval split contract (out-of-sample honesty)
# ======================================================================
_split = pc.evaluate_train_eval_split(_Req, train_frac=0.7, candidates=["equal_weight", "risk_parity"],
                                      n_paths=500, stability_resamples=0)
check("train/eval split: train_obs + eval_obs == total observations",
      _split["train_obs"] + _split["eval_obs"] == _Req.shape[0],
      f"{_split['train_obs']}+{_split['eval_obs']} vs {_Req.shape[0]}")
check("train/eval split: every evaluated row is out-of-sample",
      all(r["evaluation"] == "out-of-sample" for r in _split["rows"]))
# weights must be fit ONLY on the training slice
_split_idx = _split["train_obs"]
_train_w = pc.candidate_weights(_Req[:_split_idx], ["risk_parity"])["risk_parity"]
_row_rp = next(r for r in _split["rows"] if r["candidate"] == "risk_parity")
check("train/eval split: candidate weights are fit on the train slice only",
      np.allclose(_row_rp["weights"], _train_w, atol=1e-10))

# ======================================================================
# SECTION 12 -- input validation (must raise, not silently mislead)
# ======================================================================
def _raises(fn):
    try:
        fn(); return False
    except Exception:
        return True
_good = exact_cov_returns(np.diag([0.01, 0.02]) ** 2, n_obs=200, seed=20)
check("validation: NaN in returns raises",
      _raises(lambda: pc.risk_parity_weights(np.array([[0.1, np.nan], [0.2, 0.3]]))))
check("validation: 1-D returns matrix raises",
      _raises(lambda: pc.covariance_matrix(np.array([0.1, 0.2, 0.3]))))
check("validation: CVaR alpha outside (0,1) raises",
      _raises(lambda: pc.min_cvar_weights(_good, alpha=1.5)))
check("validation: mismatched fit/eval asset counts raises",
      _raises(lambda: pc.evaluate_candidates(_good, eval_returns=exact_cov_returns(
          np.diag([0.01, 0.02, 0.03]) ** 2, n_obs=200, seed=21), n_paths=100, stability_resamples=0)))
check("validation: train_frac out of bounds raises",
      _raises(lambda: pc.evaluate_train_eval_split(_good, train_frac=0.95)))
check("validation: non-positive risk_aversion raises",
      _raises(lambda: pc.mean_variance_weights(_good, risk_aversion=0.0)))

# ======================================================================
# REPORT
# ======================================================================
print("=" * 74)
print("CONSTRUCTION-LAB QA  (closed-form / statistical invariants, deterministic)")
print(f"SciPy exact-LP path: {'ENABLED' if HAVE_SCIPY else 'SKIPPED (optional dep not installed)'}")
print("=" * 74)
for status, name, detail in results:
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
n_fail = sum(1 for s, _, _ in results if s == FAIL)
n_skip = sum(1 for s, _, _ in results if s == SKIP)
n_pass = sum(1 for s, _, _ in results if s == PASS)
print("-" * 74)
tail = "" if n_fail == 0 else f"   ({n_fail} FAILED)"
tail += "" if n_skip == 0 else f"   ({n_skip} skipped)"
print(f"  {n_pass}/{n_pass + n_fail} checks passed{tail}")
print("=" * 74)
import sys
sys.exit(1 if n_fail else 0)
