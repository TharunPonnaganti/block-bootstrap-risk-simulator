"""
Portfolio construction helpers for candidate allocations.

This module proposes long-only candidate weights. It does not recommend a
portfolio; candidates should be evaluated by the bootstrap engine before use.
The first pass is intentionally NumPy-only: equal weight, inverse volatility,
and full equal-risk-contribution risk parity.
"""

import numpy as np

import stock_probability_engine as spe


DEFAULT_CANDIDATES = ("equal_weight", "inverse_vol", "risk_parity", "min_cvar", "mean_variance")


def _as_returns_matrix(returns_matrix):
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be a 2-D array of shape observations x assets.")
    if R.shape[0] < 2 or R.shape[1] < 1:
        raise ValueError("returns_matrix needs at least 2 observations and 1 asset.")
    if not np.isfinite(R).all():
        raise ValueError("returns_matrix contains NaN or infinite values.")
    return R


def _asset_count(assets):
    if isinstance(assets, int):
        n_assets = assets
    else:
        if isinstance(assets, str):
            raise ValueError("assets must be an int count or a sequence of asset labels.")
        n_assets = len(list(assets))
    if n_assets < 1:
        raise ValueError("asset count must be positive.")
    return n_assets


def covariance_matrix(returns_matrix, ddof=1):
    """Sample covariance matrix with stable shape for single/multi-asset inputs."""
    R = _as_returns_matrix(returns_matrix)
    if R.shape[0] <= ddof:
        raise ValueError("not enough observations for requested covariance ddof.")
    centered = R - R.mean(axis=0)
    return centered.T @ centered / (R.shape[0] - ddof)


def equal_weight(assets):
    """Return equal long-only weights for an asset count or ticker sequence."""
    n_assets = _asset_count(assets)
    return np.full(n_assets, 1.0 / n_assets, dtype=float)


def inverse_vol_weights(returns_matrix, ddof=1):
    """Return long-only weights proportional to inverse realized volatility."""
    R = _as_returns_matrix(returns_matrix)
    vol = R.std(axis=0, ddof=ddof)
    if not np.isfinite(vol).all() or np.any(vol <= 0):
        raise ValueError("all assets must have positive finite volatility.")
    inv = 1.0 / vol
    return inv / inv.sum()


def risk_contributions(weights, cov):
    """Return each asset's fraction of total portfolio variance contribution."""
    w = np.asarray(weights, dtype=float)
    S = np.asarray(cov, dtype=float)
    if w.ndim != 1 or S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != w.size:
        raise ValueError("weights and covariance dimensions do not align.")
    if not np.isfinite(w).all() or not np.isfinite(S).all():
        raise ValueError("weights and covariance must be finite.")
    cov_w = S @ w
    port_var = float(w @ cov_w)
    if port_var <= 0 or not np.isfinite(port_var):
        raise ValueError("portfolio variance must be positive.")
    return w * cov_w / port_var


def risk_parity_weights(returns_matrix, max_iter=5000, tol=1e-10, ridge=1e-12):
    """Full equal-risk-contribution risk parity weights.

    Solves the long-only ERC problem with cyclic coordinate descent on:
        0.5 * x.T @ Sigma @ x - sum_i b_i * log(x_i)
    then normalizes x to weights. At convergence, each asset contributes 1/n
    of total variance risk.
    """
    S = covariance_matrix(returns_matrix)
    S = 0.5 * (S + S.T)
    diag = np.diag(S)
    if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
        raise ValueError("all assets must have positive finite variance.")
    avg_diag = float(diag.mean())
    S = S + np.eye(S.shape[0]) * max(ridge * avg_diag, 0.0)

    n_assets = S.shape[0]
    budget = np.full(n_assets, 1.0 / n_assets)
    x = 1.0 / np.sqrt(np.diag(S))
    cov_x = S @ x

    for _ in range(max_iter):
        for i in range(n_assets):
            old = x[i]
            c_i = cov_x[i] - S[i, i] * old
            new = (-c_i + np.sqrt(c_i * c_i + 4.0 * S[i, i] * budget[i])) / (2.0 * S[i, i])
            delta = new - old
            if delta:
                x[i] = new
                cov_x += S[:, i] * delta

        w = x / x.sum()
        rc = risk_contributions(w, S)
        if np.max(np.abs(rc - budget)) < tol:
            return w

    raise RuntimeError("risk parity solver did not converge.")


def hhi(weights):
    """Herfindahl-Hirschman concentration index for portfolio weights."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all():
        raise ValueError("weights must be a finite 1-D array.")
    return float(np.sum(w * w))


def effective_number_of_bets(weights):
    """Effective number of equally sized bets implied by HHI."""
    concentration = hhi(weights)
    if concentration <= 0:
        raise ValueError("HHI must be positive.")
    return float(1.0 / concentration)


def _project_simplex(v):
    """Euclidean projection onto the long-only unit simplex."""
    x = np.asarray(v, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("simplex projection input must be a finite 1-D array.")
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, x.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return equal_weight(x.size)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(x - theta, 0.0)


def empirical_cvar_loss(returns_matrix, weights, alpha=0.95):
    """Empirical CVaR of portfolio losses, where loss = -portfolio return."""
    R = _as_returns_matrix(returns_matrix)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != R.shape[1]:
        raise ValueError("weights must align with returns_matrix columns.")
    losses = -(R @ w)
    cutoff = np.quantile(losses, alpha)
    tail = losses >= cutoff
    if not np.any(tail):
        return float(cutoff)
    return float(losses[tail].mean())


def _min_cvar_weights_linprog(R, alpha):
    """Exact Rockafellar-Uryasev CVaR LP using scipy when available."""
    try:
        from scipy.optimize import linprog
    except Exception as exc:
        raise ImportError("scipy is not installed") from exc

    n_obs, n_assets = R.shape
    tail_scale = 1.0 / ((1.0 - alpha) * n_obs)
    n_vars = n_assets + 1 + n_obs  # weights, eta, slacks

    c = np.zeros(n_vars)
    c[n_assets] = 1.0
    c[n_assets + 1:] = tail_scale

    # u_i >= -r_i @ w - eta  ->  -r_i @ w - eta - u_i <= 0
    A_ub = np.zeros((n_obs, n_vars))
    A_ub[:, :n_assets] = -R
    A_ub[:, n_assets] = -1.0
    A_ub[:, n_assets + 1:] = -np.eye(n_obs)
    b_ub = np.zeros(n_obs)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n_assets + [(None, None)] + [(0.0, None)] * n_obs
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Min-CVaR LP failed: {res.message}")
    return _project_simplex(res.x[:n_assets])


def _min_cvar_weights_subgradient(R, alpha, max_iter, tol, seed):
    """Pure-NumPy projected subgradient fallback for empirical CVaR loss."""
    rng = np.random.default_rng(seed)
    n_obs, n_assets = R.shape
    starts = [
        equal_weight(n_assets),
        inverse_vol_weights(R),
        rng.dirichlet(np.ones(n_assets)),
    ]
    best_w = starts[0]
    best_obj = empirical_cvar_loss(R, best_w, alpha)
    scale = max(float(np.std(R)), 1e-8)
    base_step = 0.8 / scale
    tail_n = max(1, int(np.ceil((1.0 - alpha) * n_obs)))

    for start in starts:
        w = start.copy()
        prev_obj = empirical_cvar_loss(R, w, alpha)
        for it in range(1, max_iter + 1):
            losses = -(R @ w)
            tail_idx = np.argpartition(losses, -tail_n)[-tail_n:]
            grad = -R[tail_idx].mean(axis=0)
            step = base_step / np.sqrt(it)
            w = _project_simplex(w - step * grad)
            obj = empirical_cvar_loss(R, w, alpha)
            if obj < best_obj:
                best_obj, best_w = obj, w.copy()
            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj
    return best_w


def min_cvar_weights(returns_matrix, alpha=0.95, solver="auto",
                     max_iter=1000, tol=1e-10, seed=42, return_info=False):
    """Long-only weights that minimize empirical portfolio loss CVaR.

    `solver="auto"` uses the exact Rockafellar-Uryasev LP if SciPy is installed,
    otherwise it falls back to a pure-NumPy projected subgradient method. Pass
    `return_info=True` to expose which solver was used.
    """
    R = _as_returns_matrix(returns_matrix)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    solver_used = solver
    if solver in ("auto", "linprog"):
        try:
            w = _min_cvar_weights_linprog(R, alpha)
            solver_used = "linprog"
        except ImportError:
            if solver == "linprog":
                raise
            w = _min_cvar_weights_subgradient(R, alpha, max_iter, tol, seed)
            solver_used = "subgradient"
    elif solver == "subgradient":
        w = _min_cvar_weights_subgradient(R, alpha, max_iter, tol, seed)
    else:
        raise ValueError("solver must be 'auto', 'linprog', or 'subgradient'.")
    if return_info:
        return w, {
            "solver": solver_used,
            "alpha": alpha,
            "cvar_loss": empirical_cvar_loss(R, w, alpha),
        }
    return w


def mean_variance_weights(returns_matrix, risk_aversion=10.0, max_iter=3000, tol=1e-12):
    """Long-only mean-variance utility candidate on the unit simplex.

    Minimizes 0.5 * risk_aversion * w.T @ Sigma @ w - mu.T @ w.
    It is a baseline candidate, not a recommendation.
    """
    R = _as_returns_matrix(returns_matrix)
    if risk_aversion <= 0 or not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be positive and finite.")
    mu = R.mean(axis=0)
    S = covariance_matrix(R)
    eig_max = float(np.linalg.eigvalsh(S).max())
    step = 1.0 / max(risk_aversion * eig_max, 1e-8)
    w = equal_weight(R.shape[1])
    for _ in range(max_iter):
        old = w
        grad = risk_aversion * (S @ w) - mu
        w = _project_simplex(w - step * grad)
        if np.linalg.norm(w - old, ord=1) < tol:
            break
    return w


def resampled_efficient_frontier(returns_matrix, risk_aversions=(2.0, 5.0, 10.0, 20.0),
                                 n_resamples=25, block=21, seed=42):
    """Mean-variance baseline refit over bootstrap resamples.

    Returns rows for a fuzzy frontier cloud plus per-risk-aversion weight
    dispersion summaries. This is a fragility diagnostic, not an optimizer verdict.
    """
    R = _as_returns_matrix(returns_matrix)
    rng = np.random.default_rng(seed)
    rows = []
    for ra in risk_aversions:
        weights = []
        for _ in range(n_resamples):
            idx = spe._block_indices([R.shape[0]], [1.0], R.shape[0], 1, block, rng)[0]
            w = mean_variance_weights(R[idx], risk_aversion=ra)
            mu = float((R[idx].mean(axis=0) @ w) * 252.0)
            vol = float(np.sqrt(w @ covariance_matrix(R[idx]) @ w) * np.sqrt(252.0))
            rows.append({
                "risk_aversion": float(ra),
                "ann_return": mu,
                "ann_vol": vol,
                "weights": w.tolist(),
            })
            weights.append(w)
        W = np.vstack(weights)
        for row in rows[-n_resamples:]:
            row["weight_std_mean"] = float(W.std(axis=0, ddof=1).mean()) if n_resamples > 1 else 0.0
            row["weight_std_max"] = float(W.std(axis=0, ddof=1).max()) if n_resamples > 1 else 0.0
    return rows


def candidate_weights(fit_returns, candidates=None):
    """Build named candidate allocations from the fitting return matrix."""
    R = _as_returns_matrix(fit_returns)
    names = DEFAULT_CANDIDATES if candidates is None else tuple(candidates)
    out = {}
    for name in names:
        if name == "equal_weight":
            out[name] = equal_weight(R.shape[1])
        elif name == "inverse_vol":
            out[name] = inverse_vol_weights(R)
        elif name == "risk_parity":
            out[name] = risk_parity_weights(R)
        elif name == "min_cvar":
            out[name] = min_cvar_weights(R)
        elif name == "mean_variance":
            out[name] = mean_variance_weights(R)
        else:
            raise ValueError(f"Unknown candidate: {name}")
    return out


def _weight_stability(fit_returns, candidate, n_resamples, block, rng):
    R = _as_returns_matrix(fit_returns)
    if n_resamples <= 0:
        return None
    samples = []
    for _ in range(n_resamples):
        idx = spe._block_indices([R.shape[0]], [1.0], R.shape[0], 1, block, rng)[0]
        try:
            samples.append(candidate_weights(R[idx], [candidate])[candidate])
        except (ValueError, RuntimeError):
            continue
    if not samples:
        return None
    W = np.vstack(samples)
    return {
        "weight_std_mean": float(W.std(axis=0, ddof=1).mean()) if W.shape[0] > 1 else 0.0,
        "weight_std_max": float(W.std(axis=0, ddof=1).max()) if W.shape[0] > 1 else 0.0,
        "weight_p10": np.percentile(W, 10, axis=0).tolist(),
        "weight_p90": np.percentile(W, 90, axis=0).tolist(),
        "n_resamples": int(W.shape[0]),
    }


def evaluate_candidates(fit_returns, eval_returns=None, candidates=None, years=5,
                        n_paths=2000, block=21, amount=10_000.0, cash_rate=0.04,
                        var_conf=0.95, stability_resamples=25, seed=42):
    """Build and bootstrap-evaluate candidate allocations.

    Candidates are fit on `fit_returns`. If `eval_returns` is omitted, evaluation
    also uses `fit_returns` and every optimized row is labeled "(in-sample)".
    Passing `eval_returns` switches the label to "out-of-sample" without changing
    the public contract.
    """
    fit_R = _as_returns_matrix(fit_returns)
    eval_R = fit_R if eval_returns is None else _as_returns_matrix(eval_returns)
    if eval_R.shape[1] != fit_R.shape[1]:
        raise ValueError("fit_returns and eval_returns must have the same asset count.")

    eval_label = "in-sample" if eval_returns is None else "out-of-sample"
    suffix = " (in-sample)" if eval_returns is None else ""
    rng = np.random.default_rng(seed)
    n_periods = int(round(years * 252))
    weights_by_name = candidate_weights(fit_R, candidates)

    rows = []
    for name, w in weights_by_name.items():
        vp = spe.bootstrap_portfolio(eval_R, w, n_periods, n_paths, block, rng, amount=amount)
        metrics = spe.analyze(vp, years, amount=amount, cash_rate=cash_rate, var_conf=var_conf)
        row = {
            "candidate": name,
            "label": f"{name}{suffix}",
            "evaluation": eval_label,
            "weights": w.tolist(),
            "hhi": hhi(w),
            "effective_bets": effective_number_of_bets(w),
        }
        row.update(metrics)
        stability = _weight_stability(fit_R, name, stability_resamples, block, rng)
        if stability is not None:
            row.update(stability)
        rows.append(row)
    return rows


def evaluate_train_eval_split(returns_matrix, train_frac=0.70, **kwargs):
    """Run the candidate lab on a chronological train/eval split.

    Weights are fit on the first `train_frac` share of observations and evaluated
    on the remaining observations. This exercises the out-of-sample hook directly.
    """
    R = _as_returns_matrix(returns_matrix)
    if not (0.2 <= train_frac <= 0.9):
        raise ValueError("train_frac must be between 0.2 and 0.9.")
    split = int(round(R.shape[0] * train_frac))
    if split < 2 or R.shape[0] - split < 2:
        raise ValueError("train/eval split leaves too few observations.")
    rows = evaluate_candidates(R[:split], eval_returns=R[split:], **kwargs)
    return {
        "train_obs": int(split),
        "eval_obs": int(R.shape[0] - split),
        "train_frac": float(train_frac),
        "rows": rows,
    }
