"""
STOCK PROBABILITY ENGINE -- local Streamlit UI
==============================================
A fully-local front end over stock_probability_engine.py. Nothing leaves your
PC except the price-history fetch from Yahoo (or use your own CSV).

RUN:
    pip install streamlit
    streamlit run app.py
Then it opens in your browser at http://localhost:8501

Iterate freely with Claude Code -- this file is just a thin view layer; all the
math lives in (and stays validated in) stock_probability_engine.py.
"""
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import stock_probability_engine as spe

st.set_page_config(page_title="Stock Probability Engine", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------
# Engine call -- cached on everything EXCEPT threshold, so dragging the
# threshold slider only re-evaluates the indicator flag (instant), never re-simulates.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching history and running the bootstrap…")
def run_engine(ticker, csv_bytes, years, blend, amount, paths, haircut, portfolio=None):
    csv_path = None
    if csv_bytes is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(csv_bytes)
        tmp.close()
        csv_path = tmp.name
    args = SimpleNamespace(
        ticker=(ticker or "VTI"), csv=csv_path, years=years, blend=blend,
        portfolio=portfolio, threshold=0.5, amount=float(amount), paths=int(paths),
        haircut=float(haircut), json=False,
    )
    return spe.compute(args)


def above_threshold(p_profit, threshold):
    return p_profit >= threshold


# ----------------------------------------------------------------------
# Sidebar -- inputs
# ----------------------------------------------------------------------
st.sidebar.title("Inputs")

data_mode = st.sidebar.radio(
    "Data", ["Single ticker (fund or stock)", "Portfolio (multi-asset)", "Upload CSV"], index=0)
ticker, csv_bytes, portfolio = None, None, None
if data_mode == "Single ticker (fund or stock)":
    TICKER_CHOICES = [
        "— diversified funds (sound prior) —",
        "VTI — US total market", "VOO — S&P 500", "SPY — S&P 500",
        "QQQ — Nasdaq-100", "DIA — Dow 30", "IWM — US small-cap",
        "SCHD — US dividend", "VXUS — ex-US total", "VWO — emerging markets",
        "VT — world all-cap", "BND — US bonds", "AGG — US bonds",
        "— individual companies (weaker prior) —",
        "AAPL — Apple", "MSFT — Microsoft", "NVDA — Nvidia", "GOOGL — Alphabet",
        "AMZN — Amazon", "META — Meta", "TSLA — Tesla", "NFLX — Netflix",
        "AMD — AMD", "JPM — JPMorgan Chase", "V — Visa", "WMT — Walmart",
        "KO — Coca-Cola", "DIS — Disney", "JNJ — Johnson & Johnson",
        "XOM — ExxonMobil", "BRK-B — Berkshire Hathaway",
        "Other (type a ticker)…",
    ]
    pick = st.sidebar.selectbox(
        "Ticker", TICKER_CHOICES, index=1,                 # default to VTI
        help="Diversified funds (top) are the statistically sound use. Individual companies "
             "run too, but earn a weaker-prior caveat. Pick 'Other' to type any symbol.")
    if pick.startswith("Other"):
        ticker = st.sidebar.text_input("Custom ticker", "VTI").strip().upper()
    elif pick.startswith("—"):                              # a section header was selected
        st.sidebar.info("Pick a ticker below the section header.")
        st.stop()
    else:
        ticker = pick.split(" ")[0].strip().upper()
elif data_mode == "Portfolio (multi-asset)":
    portfolio = st.sidebar.text_input(
        "Allocation", "VTI:0.8, QQQ:0.2",
        help="Weights are normalized. Components are date-aligned and resampled JOINTLY, "
             "so cross-asset correlation is preserved.").strip()
else:
    up = st.sidebar.file_uploader("CSV with a date + close/adj-close column", type=["csv"])
    if up is not None:
        csv_bytes = up.getvalue()

st.sidebar.markdown("---")
years, blend = None, False
if data_mode == "Portfolio (multi-asset)":
    st.sidebar.caption("Portfolio mode uses the full date-aligned history with joint resampling.")
else:
    window_mode = st.sidebar.radio(
        "History window", ["Blend eras (recommended)", "Full history", "Last N years"], index=0,
        help="Blend mixes return blocks across recent + old eras so the result isn't "
             "locked into one regime. Full history is harshest on volatility; a short "
             "recent window risks being all-bull.")
    if window_mode == "Blend eras (recommended)":
        blend = True
        st.sidebar.caption(f"Blend weights: {spe.BLEND}")
    elif window_mode == "Last N years":
        years = st.sidebar.slider("Years of history", 1, 30, 15)

st.sidebar.markdown("---")
amount = st.sidebar.number_input("Invest amount ($)", 100, 10_000_000, 10_000, step=500)
threshold = st.sidebar.slider("Flag P(profit) ≥", 0.50, 0.95, 0.70, 0.01)
haircut = st.sidebar.slider("Drift haircut (stress test)", 0.0, 1.0, 0.0, 0.05,
                            help="Shave this fraction of the historical drift to "
                                 "stress a more conservative view. 0 = use real history.")
paths = st.sidebar.select_slider("Bootstrap paths", [2000, 5000, 10000, 20000], value=10000)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
st.title("📈 Block-Bootstrap Portfolio Risk Simulator")
st.caption("Block-bootstrap of real return history → outcome distribution, tail risk "
           "(VaR/CVaR), drawdown, and P(profit). Forecasts are calibrated out-of-sample "
           "(see calibration.py). **Research tool — not investment advice.**")

try:
    res = run_engine(ticker, csv_bytes, years, blend, amount, paths, haircut, portfolio)
except Exception as e:
    st.error(f"Could not run the engine: {e}")
    st.info("If fetching by ticker failed, check the symbol or try again — the public "
            "data endpoint occasionally rate-limits. Or switch to **Upload CSV**.")
    st.stop()

h, p = res["history"], res["params"]

# ---- history / prior diagnostics -------------------------------------
st.subheader("The prior — what this stock has actually done")
c = st.columns(5)
c[0].metric("History", f"{h['years']:.1f} yrs", f"{h['obs']:,} obs")
c[1].metric("Annualized drift", f"{h['drift']*100:.1f}%")
c[2].metric("Annualized vol", f"{h['vol']*100:.1f}%")
c[3].metric("Worst day", f"{h['worst_day']*100:.1f}%")
c[4].metric("Max drawdown", f"{h['max_drawdown']*100:.1f}%")
st.caption(f"Source: {res['source']}  ·  sampling window: **{res['mode']}**  ·  "
           f"block {p['block']} · {p['paths']:,} paths"
           + (f" · drift haircut {p['haircut']*100:.0f}%" if p['haircut'] else ""))

# window composition (blend or capped window)
if res["mode"] == "blended" or len(res["windows"]) > 1 or res["windows"][0]["years"] is not None:
    comp_rows = [{
        "window": ("full" if w["years"] is None else f"{w['years']}y"),
        "weight": f"{w['weight']*100:.0f}%", "obs": w["obs"],
        "drift": f"{w['drift']*100:+.1f}%", "vol": f"{w['vol']*100:.0f}%",
    } for w in res["windows"]]
    with st.expander("Window composition feeding the simulation", expanded=(res["mode"] == "blended")):
        st.table(pd.DataFrame(comp_rows))

for w in res["warnings"]:
    (st.warning if not w.startswith("BLEND") else st.info)(w)

st.markdown("---")

# ---- P(profit)-vs-threshold indicator cards (threshold applied live) -
st.subheader(f"P(profit) vs. your threshold ({threshold*100:.0f}%)")
st.caption("A mechanical indicator, **not** a buy/sell signal — calibration shows P(profit) "
           "has no demonstrated skill vs a base rate. Read the risk distribution below it.")
cols = st.columns(len(res["horizons"]))
for col, r in zip(cols, res["horizons"]):
    above = above_threshold(r["P(profit)"], threshold)
    with col:
        flag = "above threshold" if above else "below threshold"
        if above:
            st.success(f"### {r['years']}-year hold\n#### P(profit) {flag}")
        else:
            st.info(f"### {r['years']}-year hold\n#### P(profit) {flag}")
        st.metric("P(profit)", f"{r['P(profit)']*100:.1f}%")
        st.metric(f"Median ${amount:,.0f} →", f"${r['val_P50']:,.0f}",
                  f"{(r['val_P50']/amount-1)*100:+.0f}%")
        st.caption(
            f"Range P10–P90: ${r['val_P10']:,.0f} → ${r['val_P90']:,.0f}\n\n"
            f"Beats cash@{p['cash_rate']*100:.0f}%: {r['P(beat cash)']*100:.0f}%\n\n"
            f"Worst-5% end: {r['var_ret']*100:+.0f}% (CVaR {r['cvar_ret']*100:+.0f}%)\n\n"
            f"Drawdown: median {r['maxdd_med']*100:.0f}%, bad-case {r['maxdd_p95worst']*100:.0f}%"
        )

st.markdown("---")

# ---- probability cone ------------------------------------------------
st.subheader(f"Probability cone — ${amount:,.0f} over {max(spe.HORIZONS_YEARS)} years")
fan = res["fan"]
df = pd.DataFrame(fan)
base = alt.Chart(df).encode(x=alt.X("years:Q", title="Years held"))
bands = [
    ("p5", "p95", 0.12), ("p10", "p90", 0.18), ("p25", "p75", 0.30),
]
layers = []
for lo, hi, op in bands:
    layers.append(
        base.mark_area(opacity=op, color="#2e7d32").encode(
            y=alt.Y(f"{lo}:Q", title="Portfolio value ($)"), y2=f"{hi}:Q")
    )
layers.append(base.mark_line(color="#1b5e20", strokeWidth=2.5).encode(y="p50:Q"))
invested_line = alt.Chart(pd.DataFrame({"y": [amount]})).mark_rule(
    color="gray", strokeDash=[5, 5]).encode(y="y:Q")
st.altair_chart(alt.layer(*layers, invested_line).properties(height=380).interactive(),
                use_container_width=True)
st.caption("Shaded bands = 90% / 80% / 50% of simulated outcomes. Solid line = median. "
           "Dashed line = amount invested (above it = profit).")

with st.expander("How to read this / limitations"):
    st.markdown(
        "- **P(profit)** = share of simulated futures ending above what you put in.\n"
        "- **VaR/CVaR** = the bad tail: where you end in the worst 5%, and its average.\n"
        "- **Drawdown** = worst peak-to-trough dip you'd sit through, even on good paths.\n"
        "- The threshold flag is a **mechanical comparison on a historical prior — not advice "
        "and not a trading signal** (calibration shows P(profit) has no demonstrated skill vs a "
        "base rate). A single stock can break from its history (earnings, fraud, obsolescence) in "
        "ways no bootstrap can see. All figures are **nominal** (not inflation-adjusted)."
    )
