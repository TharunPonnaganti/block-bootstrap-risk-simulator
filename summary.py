"""Plain-English summary of a simulation run.

Turns the results dict from stock_probability_engine.compute() into a short
markdown narrative written for non-experts: natural frequencies ("95 out of
100 futures") instead of percentiles, no method jargon in the body.
Deterministic template filling -- every number comes straight from the
engine's output, no generation, no external calls.
"""


def _money(val, sym):
    return f"{sym}{val:,.0f}"


def _pct(x, signed=False):
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


def _freq(prob):
    """0.945 -> '95 out of 100' (natural frequency, easier for lay readers)."""
    return f"{round(prob * 100)} out of 100"


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

    # ---- what this is -------------------------------------------------
    haircut_note = (f" As a stress test, {p['haircut'] * 100:.0f}% of the historical growth "
                    f"trend was removed before simulating."
                    if p.get("haircut") else "")
    parts.append(
        f"**What this is:** we took {name}'s last {h['years']:.1f} years of real market "
        f"movements ({h['obs']:,} trading days) and replayed them in {p['paths']:,} different "
        f"orders to see how an investment could play out. Real history, rearranged — so the "
        f"crashes and wild stretches stay in the picture.{haircut_note}")

    # ---- the focus horizon, lump sum -----------------------------------
    parts.append(
        f"**If you invest {_money(amount, sym)} today and hold {focus['years']} years:** "
        f"in {_freq(focus['P(profit)'])} of these simulated futures, you end up with more than "
        f"you put in. Half end above {_money(focus['val_P50'], sym)}, half below. There is "
        f"about a 1-in-10 chance of ending below {_money(focus['val_P10'], sym)}, and the "
        f"unluckiest 1-in-20 futures finish at {_pct(focus['var_ret'], signed=True)} or worse "
        f"(that group averages {_pct(focus['cvar_ret'], signed=True)}).")

    # ---- the ride along the way -----------------------------------------
    parts.append(
        f"**The ride along the way:** even futures that end well rarely travel smoothly. The "
        f"typical one dips {_pct(-focus['maxdd_med'])} below its high point at some stage before "
        f"recovering; the rough ones dip {_pct(-focus['maxdd_p95worst'])}. Whether you could sit "
        f"through a dip like that without selling is the real question.")

    # ---- how the horizon changes the odds --------------------------------
    if len(horizons) >= 2:
        first = horizons[0]
        d = focus["P(profit)"] - first["P(profit)"]
        verb = "improves" if d > 0.005 else ("worsens" if d < -0.005 else "barely changes")
        yr1 = f"{first['years']} year" + ("s" if first["years"] != 1 else "")
        parts.append(
            f"**Holding longer:** in this history, patience {verb} the odds — "
            f"{_freq(first['P(profit)'])} futures end in profit after {yr1}, vs "
            f"{_freq(focus['P(profit)'])} after {focus['years']} years.")

    # ---- monthly plan (DCA/SIP), if enabled ------------------------------
    dca = res.get("dca")
    if dca and dca.get("horizons"):
        fd = dca["horizons"][-1]
        parts.append(
            f"**Investing monthly instead:** {_money(dca['contrib'], sym)}/month adds up to "
            f"{_money(fd['total_invested'], sym)} over {fd['years']} years "
            f"({fd['n_contributions']} contributions). In {_freq(fd['P(profit)'])} of the same "
            f"futures you end above that total; half end above {_money(fd['val_P50'], sym)} "
            f"({_pct(fd['total_return_P50'], signed=True)} on what you put in). Same markets — "
            f"the difference is when the money goes in.")

    # ---- warnings, verbatim ------------------------------------------------
    if res.get("warnings"):
        parts.append("**Heads up — this run raised:**\n" +
                     "\n".join(f"- {w}" for w in res["warnings"]))

    # ---- method & data footer -----------------------------------------------
    parts.append(
        f"**Where this comes from:** {res['source']}. The method replays month-sized chunks of "
        f"real market history (a \"block bootstrap\", Politis & Romano 1992), so real crashes "
        f"and streaks stay in — nothing is assumed to follow a smooth bell curve. The cash "
        f"comparison uses {p['cash_rate'] * 100:.1f}%/yr. Values are not adjusted for inflation. "
        f"These probabilities describe the simulation, not the future. Not investment advice.")

    return "\n\n".join(parts)
