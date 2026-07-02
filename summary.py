"""Plain-English summary of a simulation run.

Turns the results dict from stock_probability_engine.compute() into a short
markdown narrative. Deterministic template filling -- every number comes
straight from the engine's output, no generation, no external calls.
"""


def _money(val, sym):
    return f"{sym}{val:,.0f}"


def _pct(x, signed=False):
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


def summarize(res):
    """Return a markdown summary of a compute() results dict."""
    h = res["history"]
    p = res["params"]
    sym = res.get("currency", {}).get("symbol", "$")
    amount = float(p["amount"])
    horizons = res["horizons"]
    focus = horizons[-1]                      # longest horizon
    name = res["source"].split("|")[-1].strip()

    parts = []

    # ---- what you're looking at -------------------------------------
    haircut_note = (f", with {p['haircut'] * 100:.0f}% of historical drift removed as a stress test"
                    if p.get("haircut") else "")
    parts.append(
        f"**What you're looking at:** {name} — {h['years']:.1f} years of real history "
        f"({h['obs']:,} trading days), reshuffled into {p['paths']:,} simulated futures "
        f"with a circular block bootstrap{haircut_note}.")

    # ---- the focus horizon, lump sum --------------------------------
    parts.append(
        f"**A {_money(amount, sym)} lump sum over {focus['years']} years:** "
        f"{_pct(focus['P(profit)'])} of simulated futures ended above what you invested. "
        f"The median path reaches {_money(focus['val_P50'], sym)}; the pessimistic tenth ends at or "
        f"below {_money(focus['val_P10'], sym)}. The worst 5% of outcomes finish {_pct(focus['var_ret'], signed=True)} "
        f"or lower (that tail averages {_pct(focus['cvar_ret'], signed=True)}). Along the way, the typical path "
        f"dips {_pct(focus['maxdd_med'])} from a peak at some point — in the bad case, "
        f"{_pct(focus['maxdd_p95worst'])}. That dip is what you would have to sit through.")

    # ---- how the horizon changes the odds ----------------------------
    if len(horizons) >= 2:
        first = horizons[0]
        d = focus["P(profit)"] - first["P(profit)"]
        trend = "rise" if d > 0.005 else ("fall" if d < -0.005 else "stay roughly flat")
        parts.append(
            f"**Time in the market:** the odds of profit {trend} with holding period here — "
            f"{_pct(first['P(profit)'])} at {first['years']} year(s) vs {_pct(focus['P(profit)'])} at "
            f"{focus['years']} years, based on this history.")

    # ---- monthly plan (DCA/SIP), if enabled --------------------------
    dca = res.get("dca")
    if dca and dca.get("horizons"):
        fd = dca["horizons"][-1]
        parts.append(
            f"**Adding {_money(dca['contrib'], sym)}/month instead:** over {fd['years']} years that is "
            f"{fd['n_contributions']} contributions totalling {_money(fd['total_invested'], sym)}. "
            f"{_pct(fd['P(profit)'])} of the same market scenarios end above that total; the median "
            f"portfolio reaches {_money(fd['val_P50'], sym)} ({_pct(fd['total_return_P50'], signed=True)} on invested). "
            f"Same market, different cash-flow timing.")

    # ---- warnings, verbatim ------------------------------------------
    if res.get("warnings"):
        parts.append("**Heads up — this run raised:**\n" +
                     "\n".join(f"- {w}" for w in res["warnings"]))

    # ---- method & data footer -----------------------------------------
    parts.append(
        f"**Method & data:** {res['source']}. Circular moving-block bootstrap "
        f"(Politis & Romano, 1992): blocks of ~{p['block']} trading days of real returns are "
        f"stitched into synthetic futures, so crashes, volatility clustering, and fat tails stay in — "
        f"no bell-curve assumption. Cash benchmark {p['cash_rate'] * 100:.1f}%/yr. All values nominal "
        f"(not inflation-adjusted). Probabilities describe this simulation, not the future. "
        f"Not investment advice.")

    return "\n\n".join(parts)
