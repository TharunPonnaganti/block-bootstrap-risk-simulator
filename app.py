"""
Streamlit dashboard for the block-bootstrap risk simulator.

Runs locally -- nothing leaves your machine except the Yahoo price fetch.
Start with: streamlit run app.py
"""
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import stock_probability_engine as spe
import portfolio_construction as pc

st.set_page_config(page_title="Stock Probability Engine", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------
# Currency formatting helper
# ----------------------------------------------------------------------
def fmt(val, sym):
    return f"{sym}{val:,.0f}"


# ----------------------------------------------------------------------
# Engine call -- cached on everything EXCEPT threshold
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching history and running the bootstrap...")
def run_engine(ticker, csv_bytes, years, blend, amount, paths, haircut, portfolio=None, dca=None):
    csv_path = None
    if csv_bytes is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(csv_bytes)
        tmp.close()
        csv_path = tmp.name
    args = SimpleNamespace(
        ticker=(ticker or "VTI"), csv=csv_path, years=years, blend=blend,
        portfolio=portfolio, threshold=0.5, amount=float(amount), paths=int(paths),
        haircut=float(haircut), dca=dca, dca_start="begin", json=False,
    )
    return spe.compute(args)


@st.cache_data(show_spinner="Running candidate allocation lab...")
def run_construction_lab(portfolio, amount, paths, cash_rate):
    weights = spe.parse_weights(portfolio)
    rdates, R, w, tickers, source, native_years = spe.fetch_portfolio(weights, None)
    lab_paths = max(300, min(int(paths), 1200))
    rows = pc.evaluate_candidates(
        R,
        candidates=["equal_weight", "inverse_vol", "risk_parity", "min_cvar", "mean_variance"],
        years=5,
        n_paths=lab_paths,
        block=21,
        amount=float(amount),
        cash_rate=float(cash_rate),
        stability_resamples=2,
        seed=101,
    )
    split = pc.evaluate_train_eval_split(
        R,
        train_frac=0.70,
        candidates=["equal_weight", "inverse_vol", "risk_parity", "min_cvar", "mean_variance"],
        years=1,
        n_paths=max(200, min(int(paths), 700)),
        block=21,
        amount=float(amount),
        cash_rate=float(cash_rate),
        stability_resamples=1,
        seed=202,
    )
    frontier = pc.resampled_efficient_frontier(
        R,
        risk_aversions=(2.0, 5.0, 10.0, 20.0),
        n_resamples=6,
        block=21,
        seed=303,
    )
    return {"tickers": tickers, "rows": rows, "split": split, "frontier": frontier}


def above_threshold(p_profit, threshold):
    return p_profit >= threshold


def weights_label(weights, tickers):
    return ", ".join(f"{t} {w*100:.0f}%" for t, w in zip(tickers, weights))


def candidate_table(rows, tickers, cs):
    return pd.DataFrame([{
        "Candidate": r["label"],
        "Weights": weights_label(r["weights"], tickers),
        "P(profit)": f"{r['P(profit)']*100:.1f}%",
        "P(beat cash)": f"{r['P(beat cash)']*100:.1f}%",
        "P10": fmt(r["val_P10"], cs),
        "Median": fmt(r["val_P50"], cs),
        "CVaR": f"{r['cvar_ret']*100:+.1f}%",
        "Bad-case DD": f"{r['maxdd_p95worst']*100:.1f}%",
        "Eff. bets": f"{r['effective_bets']:.2f}",
        "Weight stability": f"avg {r.get('weight_std_mean', 0)*100:.1f}% / max {r.get('weight_std_max', 0)*100:.1f}%",
    } for r in rows])


def lab_column_config():
    """Per-column hover tooltips for the candidate-allocation tables."""
    return {
        "Candidate": st.column_config.TextColumn("Candidate", help=(
            "The allocation method. equal_weight = 1/N; inverse_vol = weight proportional to "
            "1/volatility; risk_parity = full equal-risk-contribution (each asset adds the same "
            "risk); min_cvar = minimizes the worst-5% tail loss (exact LP with SciPy, else a "
            "near-optimal NumPy fallback); mean_variance = classic return-vs-variance baseline. "
            "'(in-sample)' means the weights were fit and graded on the same history.")),
        "Weights": st.column_config.TextColumn("Weights", help="The candidate's allocation across your tickers."),
        "P(profit)": st.column_config.TextColumn("P(profit)", help=(
            "Share of bootstrap futures ending above the amount invested. Uncalibrated for "
            "constructed portfolios -- read as a risk indicator, not a forecast.")),
        "P(beat cash)": st.column_config.TextColumn("P(beat cash)", help=(
            "Share of futures ending above a risk-free cash benchmark compounded over the horizon.")),
        "P10": st.column_config.TextColumn("P10", help=(
            "Pessimistic outcome: 10th-percentile ending value (90% of paths end above this).")),
        "Median": st.column_config.TextColumn("Median", help=(
            "50th-percentile ending value -- half of simulated paths land above, half below.")),
        "CVaR": st.column_config.TextColumn("CVaR", help=(
            "Conditional Value at Risk (Expected Shortfall): average return across the worst 5% "
            "of outcomes. More negative = a fatter, more painful tail.")),
        "Bad-case DD": st.column_config.TextColumn("Bad-case DD", help=(
            "Bad-case maximum drawdown: the deepest peak-to-trough fall at the 5%-worst path.")),
        "Eff. bets": st.column_config.TextColumn("Eff. bets", help=(
            "Effective number of independent bets = 1 / HHI concentration. Near the asset count "
            "= well diversified; near 1 = concentrated in a single holding.")),
        "Weight stability": st.column_config.TextColumn("Weight stability", help=(
            "How much the fitted weights move when refit on bootstrap resamples (average / max "
            "standard deviation). Higher = the optimizer is fragile/unstable on this history.")),
    }


# ----------------------------------------------------------------------
# Ticker lists by market
# ----------------------------------------------------------------------
US_TICKERS = [
    "--- US diversified funds (sound prior) ---",
    "VTI -- US total market", "VOO -- S&P 500", "SPY -- S&P 500",
    "QQQ -- Nasdaq-100", "DIA -- Dow 30", "IWM -- US small-cap",
    "SCHD -- US dividend", "VXUS -- ex-US total", "VWO -- emerging markets",
    "VT -- world all-cap", "BND -- US bonds", "AGG -- US bonds",
    "--- US individual companies (weaker prior) ---",
    "AAPL -- Apple", "MSFT -- Microsoft", "NVDA -- Nvidia", "GOOGL -- Alphabet",
    "AMZN -- Amazon", "META -- Meta", "TSLA -- Tesla", "NFLX -- Netflix",
    "AMD -- AMD", "JPM -- JPMorgan Chase", "V -- Visa", "WMT -- Walmart",
    "KO -- Coca-Cola", "DIS -- Disney", "JNJ -- Johnson & Johnson",
    "XOM -- ExxonMobil", "BRK-B -- Berkshire Hathaway",
    "Other (type a ticker)...",
]

INDIA_TICKERS = [
    "--- India index ETFs / NSE (sound prior) ---",
    "NIFTYBEES.NS -- Nifty 50 ETF", "JUNIORBEES.NS -- Nifty Next 50 ETF",
    "BANKBEES.NS -- Bank Nifty ETF", "SETFNIF50.NS -- SBI Nifty 50 ETF",
    "NIF100BEES.NS -- Nifty 100 ETF", "CPSEETF.NS -- CPSE ETF",
    "MOM100.NS -- Nifty Momentum 100 ETF",
    "--- India large-cap / NSE (weaker prior) ---",
    "RELIANCE.NS -- Reliance Industries", "TCS.NS -- TCS",
    "HDFCBANK.NS -- HDFC Bank", "INFY.NS -- Infosys",
    "ICICIBANK.NS -- ICICI Bank", "BHARTIARTL.NS -- Bharti Airtel",
    "ITC.NS -- ITC", "SBIN.NS -- State Bank of India",
    "LT.NS -- Larsen & Toubro", "KOTAKBANK.NS -- Kotak Mahindra Bank",
    "HINDUNILVR.NS -- Hindustan Unilever", "BAJFINANCE.NS -- Bajaj Finance",
    "MARUTI.NS -- Maruti Suzuki", "TATAMOTORS.NS -- Tata Motors",
    "WIPRO.NS -- Wipro", "TATASTEEL.NS -- Tata Steel",
    "ADANIENT.NS -- Adani Enterprises", "ADANIPORTS.NS -- Adani Ports",
    "HCLTECH.NS -- HCL Technologies", "SUNPHARMA.NS -- Sun Pharma",
    "TITAN.NS -- Titan Company", "POWERGRID.NS -- Power Grid Corp",
    "--- India / BSE ---",
    "NIFTYBEES.BO -- Nifty 50 ETF (BSE)", "JUNIORBEES.BO -- Nifty Next 50 (BSE)",
    "Other (type a ticker)...",
]


# ----------------------------------------------------------------------
# Sidebar -- inputs
# ----------------------------------------------------------------------
st.sidebar.title("Inputs")

market = st.sidebar.radio("Market", ["US", "India"], index=0, horizontal=True,
                          help="Sets the currency and risk-free cash rate. US uses $ and ~4%; "
                               "India uses Rs and ~6.5%, and accepts .NS (NSE) / .BO (BSE) tickers.")

data_mode = st.sidebar.radio(
    "Data", ["Single ticker (fund or stock)", "Portfolio (multi-asset)", "Upload CSV"], index=0,
    help="Single ticker = one fund or stock. Portfolio = a multi-asset mix (unlocks the candidate "
         "allocation lab). Upload CSV = use your own date + price export instead of Yahoo.")
ticker, csv_bytes, portfolio = None, None, None

if data_mode == "Single ticker (fund or stock)":
    ticker_list = US_TICKERS if market == "US" else INDIA_TICKERS
    pick = st.sidebar.selectbox(
        "Ticker", ticker_list, index=1,
        help="Diversified funds/ETFs (top) are the statistically sound use. Individual companies "
             "run too, but earn a weaker-prior caveat. Pick 'Other' to type any symbol.")
    if pick.startswith("Other"):
        default_ticker = "VTI" if market == "US" else "NIFTYBEES.NS"
        ticker = st.sidebar.text_input(
            "Custom ticker", default_ticker,
            help="Any Yahoo Finance symbol. Append .NS for NSE India or .BO for BSE India.").strip().upper()
    elif pick.startswith("---"):
        st.sidebar.info("Pick a ticker below the section header.")
        st.stop()
    else:
        ticker = pick.split(" ")[0].strip().upper()
elif data_mode == "Portfolio (multi-asset)":
    default_portfolio = "VTI:0.8, QQQ:0.2" if market == "US" else "NIFTYBEES.NS:0.6, HDFCBANK.NS:0.2, TCS.NS:0.2"
    portfolio = st.sidebar.text_input(
        "Allocation", default_portfolio,
        help="Weights are normalized. Components are date-aligned and resampled JOINTLY, "
             "so cross-asset correlation is preserved.").strip()
else:
    up = st.sidebar.file_uploader(
        "CSV with a date + close/adj-close column", type=["csv"],
        help="A brokerage export with a date column and a close/adj-close (or price/NAV) column. "
             "Columns are auto-detected. Treat a single company's CSV as a weak prior.")
    if up is not None:
        csv_bytes = up.getvalue()

# currency defaults
cur_sym = "$" if market == "US" else "₹"
default_amount = 10_000 if market == "US" else 100_000
amount_step = 500 if market == "US" else 10_000
default_dca = 500 if market == "US" else 5_000
dca_step = 100 if market == "US" else 1_000

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
        years = st.sidebar.slider("Years of history", 1, 30, 15,
                              help="Cap the lookback to the last N years. Fewer years = more "
                                   "recent regime but fewer real crashes in the sample.")

st.sidebar.markdown("---")
st.sidebar.subheader("Lump Sum")
amount = st.sidebar.number_input(
    f"Invest amount ({cur_sym})", 100, 100_000_000, default_amount, step=amount_step,
    help="Hypothetical one-time lump sum invested today. P(profit) is scale-free, but the "
         "percentile ending values scale with this amount.")

st.sidebar.subheader("Monthly Recurring (SIP / DCA)")
enable_dca = st.sidebar.checkbox(
    "Enable monthly recurring", value=True,
    help="Also simulate a monthly contribution (SIP/DCA) on the SAME market scenarios as the "
         "lump sum, so the comparison isolates cash-flow timing.")
if enable_dca:
    dca_amount = st.sidebar.number_input(
        f"Monthly contribution ({cur_sym})", 100, 10_000_000, default_dca, step=dca_step,
        help="Amount contributed at the start of each month over the horizon. Total invested "
             "grows with each contribution; P(profit) compares the ending value to that total.")
else:
    dca_amount = None

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Flag P(profit) >=", 0.50, 0.95, 0.70, 0.01,
                              help="The cutoff for the above/below flag on each horizon. "
                                   "Higher (e.g. 0.90) = stricter, so fewer horizons get "
                                   "flagged; lower (e.g. 0.55) = looser, so more do. It only "
                                   "moves the flag -- the simulation itself does not change. "
                                   "A display convenience, not a buy/sell signal.")
haircut = st.sidebar.slider("Drift haircut (stress test)", 0.0, 1.0, 0.0, 0.05,
                            help="Shave this fraction of the historical drift to stress a "
                                 "more conservative view. 0 = keep the real historical drift; "
                                 "1 = remove all drift, so paths fluctuate around flat "
                                 "(median growth approx. 0, P(profit) approx. 50%). Values "
                                 "in between partially dampen the trend.")
paths = st.sidebar.select_slider("Bootstrap paths", [2000, 5000, 10000, 20000], value=10000,
                                 help="Number of simulated futures generated. More paths give "
                                      "smoother, more stable percentiles but take longer to run.")


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
st.title("Block-Bootstrap Portfolio Risk Simulator")
st.caption("Block-bootstrap of real return history -> outcome distribution, tail risk "
           "(VaR/CVaR), drawdown, and P(profit). Forecasts are calibrated out-of-sample "
           "(see calibration.py). **Research tool -- not investment advice.**")

try:
    res = run_engine(ticker, csv_bytes, years, blend, amount, paths, haircut, portfolio,
                     dca=dca_amount)
except Exception as e:
    st.error(f"Could not run the engine: {e}")
    st.info("If fetching by ticker failed, check the symbol or try again -- the public "
            "data endpoint occasionally rate-limits. Or switch to **Upload CSV**.")
    st.stop()

h, p = res["history"], res["params"]
cs = res.get("currency", {"symbol": cur_sym, "code": "USD"})["symbol"]
dca_data = res.get("dca")

# ---- history / prior diagnostics -------------------------------------
st.subheader("The prior -- what this stock has actually done")
c = st.columns(5)
c[0].metric("History", f"{h['years']:.1f} yrs", f"{h['obs']:,} obs",
            help="Length of price history used: calendar years and number of trading-day "
                 "observations. More history means more real return blocks to resample from.")
c[1].metric("Annualized drift", f"{h['drift']*100:.1f}%",
            help="Geometric mean return, annualized, over the sampling window -- the long-run "
                 "growth rate baked into the bootstrap. Positive = the asset grew on average; "
                 "negative = it declined.")
c[2].metric("Annualized vol", f"{h['vol']*100:.1f}%",
            help="Annualized standard deviation of returns -- how much the price swings. "
                 "Higher = bigger swings and a wider outcome cone; lower = steadier and a "
                 "narrower cone.")
c[3].metric("Worst day", f"{h['worst_day']*100:.1f}%",
            help="Largest single-day loss actually observed in this history, shown as a "
                 "negative percentage (e.g. -11% means the price fell 11% in one day).")
c[4].metric("Max drawdown", f"{h['max_drawdown']*100:.1f}%",
            help="Largest peak-to-trough decline actually observed in this history -- how far "
                 "the price fell from a previous high before recovering. Shown as a negative "
                 "percentage; a deeper (more negative) value means a more painful historical "
                 "fall.")
st.caption(f"Source: {res['source']}  |  sampling window: **{res['mode']}**  |  "
           f"block {p['block']} | {p['paths']:,} paths"
           + (f" | drift haircut {p['haircut']*100:.0f}%" if p['haircut'] else ""))

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

if data_mode == "Portfolio (multi-asset)" and portfolio:
    st.markdown("---")
    st.subheader("Candidate allocation lab")
    st.caption("Objective-driven candidate allocations, not recommendations. Weights are fit on historical returns, "
               "then graded by the same block-bootstrap engine used above. Rows marked in-sample are fit and "
               "evaluated on the same return history; the train/eval table below shows a chronological split.")
    try:
        lab = run_construction_lab(portfolio, amount, paths, p["cash_rate"])
        lab_rows = lab["rows"]
        tickers_lab = lab["tickers"]
        st.dataframe(candidate_table(lab_rows, tickers_lab, cs), width="stretch",
                     hide_index=True, column_config=lab_column_config())
        st.caption("Hover the ? on any column header for what it means.")

        with st.expander("What each candidate method and metric means"):
            st.markdown(
                "**Allocation methods (each one *proposes* weights; the bootstrap then grades them):**\n"
                "- **equal_weight** -- 1/N across all assets. The naive baseline.\n"
                "- **inverse_vol** -- weight proportional to 1/volatility; calmer assets get more.\n"
                "- **risk_parity** -- full equal-risk-contribution: each asset adds the *same* risk "
                "to the portfolio (not the same capital).\n"
                "- **min_cvar** -- minimizes the worst-5% tail loss (CVaR). Exact Rockafellar-Uryasev "
                "LP if SciPy is installed, otherwise a near-optimal NumPy fallback.\n"
                "- **mean_variance** -- the classic return-vs-variance baseline (long-only).\n\n"
                "**Fragility / quality columns:**\n"
                "- **Eff. bets** -- effective number of independent bets (1/HHI). Near the asset "
                "count = diversified; near 1 = concentrated.\n"
                "- **Weight stability** -- how much the weights move when refit on resampled history. "
                "High = the optimizer is unstable and you shouldn't trust its exact weights.\n\n"
                "Rows marked **(in-sample)** are fit and graded on the same history -- optimistic by "
                "construction. The split table below grades them out-of-sample."
            )

        with st.expander("Out-of-sample split check", expanded=False):
            split = lab["split"]
            st.caption(f"Weights fit on the first {split['train_obs']:,} observations and evaluated on the "
                       f"last {split['eval_obs']:,}. This proves the lab *can* grade candidates out-of-sample; "
                       f"it is not a powered study proving any method beats the baseline.")
            st.dataframe(candidate_table(split["rows"], tickers_lab, cs), width="stretch",
                         hide_index=True, column_config=lab_column_config())

        with st.expander("Resampled mean-variance frontier cloud", expanded=False):
            fdf = pd.DataFrame(lab["frontier"])
            fdf["ann_return_pct"] = fdf["ann_return"] * 100.0
            fdf["ann_vol_pct"] = fdf["ann_vol"] * 100.0
            chart = alt.Chart(fdf).mark_circle(size=70, opacity=0.65).encode(
                x=alt.X("ann_vol_pct:Q", title="Annualized volatility (%)"),
                y=alt.Y("ann_return_pct:Q", title="Annualized mean return (%)"),
                color=alt.Color("risk_aversion:N", title="Risk aversion"),
                tooltip=["risk_aversion:N", "ann_return_pct:Q", "ann_vol_pct:Q", "weight_std_mean:Q"],
            ).properties(height=300)
            st.altair_chart(chart.interactive(), width="stretch")
            st.caption("Each point refits the mean-variance baseline on a bootstrap resample. "
                       "A wide cloud means the classic optimizer is unstable on the available history.")
    except Exception as e:
        st.warning(f"Could not run the candidate allocation lab: {e}")

st.markdown("---")

# ======================================================================
# LUMP SUM RESULTS
# ======================================================================
st.subheader(f"Lump Sum -- {fmt(amount, cs)} one-time")
st.caption(f"P(profit) vs. your threshold ({threshold*100:.0f}%) -- "
           "a mechanical indicator, **not** a buy/sell signal.")
cols = st.columns(len(res["horizons"]))
for col, r in zip(cols, res["horizons"]):
    above = above_threshold(r["P(profit)"], threshold)
    with col:
        flag = "above threshold" if above else "below threshold"
        if above:
            st.success(f"### {r['years']}-year hold\n#### P(profit) {flag}")
        else:
            st.info(f"### {r['years']}-year hold\n#### P(profit) {flag}")
        st.metric("P(profit)", f"{r['P(profit)']*100:.1f}%",
                  help="Share of simulated futures ending above what you invested, from 0% to "
                       "100%. Higher = more paths end in profit; 50% means roughly even odds. "
                       "Checked out-of-sample (see calibration.py); best read as a risk "
                       "indicator, not a directional bet.")
        st.metric(f"Median {fmt(amount, cs)} ->", fmt(r['val_P50'], cs),
                  f"{(r['val_P50']/amount-1)*100:+.0f}%",
                  help="Median (P50) ending value across all simulated paths -- half the "
                       "outcomes land above this, half below.")
        st.caption(
            f"Range P10-P90: {fmt(r['val_P10'], cs)} -> {fmt(r['val_P90'], cs)}\n\n"
            f"Beats cash@{p['cash_rate']*100:.0f}%: {r['P(beat cash)']*100:.0f}%\n\n"
            f"Worst-5% end: {r['var_ret']*100:+.0f}% (CVaR {r['cvar_ret']*100:+.0f}%)\n\n"
            f"Drawdown: median {r['maxdd_med']*100:.0f}%, bad-case {r['maxdd_p95worst']*100:.0f}%"
        )

# ======================================================================
# DCA / SIP RESULTS (side by side)
# ======================================================================
if dca_data:
    st.markdown("---")
    st.subheader(f"Monthly Recurring (SIP / DCA) -- {fmt(dca_data['contrib'], cs)}/month")
    st.caption("Same market scenarios as the lump sum above; only the cash-flow timing differs. "
               "P(profit) = portfolio value > total amount invested. "
               "**DCA probability is uncalibrated** (calibration was on lump-sum P(profit)).")
    cols = st.columns(len(dca_data["horizons"]))
    for col, rd in zip(cols, dca_data["horizons"]):
        above = above_threshold(rd["P(profit)"], threshold)
        ti = rd["total_invested"]
        with col:
            flag = "above threshold" if above else "below threshold"
            if above:
                st.success(f"### {rd['years']}-year SIP\n#### P(profit) {flag}")
            else:
                st.info(f"### {rd['years']}-year SIP\n#### P(profit) {flag}")
            st.metric("P(profit)", f"{rd['P(profit)']*100:.1f}%",
                      help="Share of simulated futures where the portfolio ends above total "
                           "invested, from 0% to 100%. Higher = more paths end in profit; 50% "
                           "means roughly even odds. Note: DCA P(profit) is uncalibrated -- "
                           "calibration was done on lump-sum P(profit).")
            st.metric(f"Total invested", fmt(ti, cs),
                      f"{rd['n_contributions']} contributions",
                      help="Sum of all monthly contributions made over this horizon.")
            st.metric(f"Median portfolio ->", fmt(rd['val_P50'], cs),
                      f"{rd['total_return_P50']*100:+.0f}% on invested",
                      help="Median (P50) portfolio value across simulated paths, and its "
                           "return measured against total amount invested.")
            st.caption(
                f"Range P10-P90: {fmt(rd['val_P10'], cs)} -> {fmt(rd['val_P90'], cs)}\n\n"
                f"Beats cash@{p['cash_rate']*100:.0f}%: {rd['P(beat cash)']*100:.0f}%\n\n"
                f"Worst-5% end: {rd['var_ret']*100:+.0f}% on invested "
                f"(CVaR {rd['cvar_ret']*100:+.0f}%)\n\n"
                f"Drawdown: median {rd['maxdd_med']*100:.0f}%, bad-case {rd['maxdd_p95worst']*100:.0f}%"
            )

st.markdown("---")

# ======================================================================
# PROBABILITY CONES -- side by side if DCA enabled
# ======================================================================
if dca_data:
    col_lump, col_dca = st.columns(2)
else:
    col_lump = st.container()

# ---- lump sum cone ---------------------------------------------------
with col_lump:
    st.subheader(f"Lump Sum Cone -- {fmt(amount, cs)}")
    fan = res["fan"]
    df = pd.DataFrame(fan)
    base = alt.Chart(df).encode(x=alt.X("years:Q", title="Years held"))
    bands = [("p5", "p95", 0.12), ("p10", "p90", 0.18), ("p25", "p75", 0.30)]
    layers = []
    for lo, hi, op in bands:
        layers.append(
            base.mark_area(opacity=op, color="#2e7d32").encode(
                y=alt.Y(f"{lo}:Q", title=f"Portfolio value ({cs})"), y2=f"{hi}:Q"))
    layers.append(base.mark_line(color="#1b5e20", strokeWidth=2.5).encode(y="p50:Q"))
    invested_line = alt.Chart(pd.DataFrame({"y": [amount]})).mark_rule(
        color="gray", strokeDash=[5, 5]).encode(y="y:Q")
    st.altair_chart(alt.layer(*layers, invested_line).properties(height=350).interactive(),
                    width="stretch")
    st.caption("Dashed line = amount invested (above it = profit).")

# ---- DCA cone --------------------------------------------------------
if dca_data:
    with col_dca:
        st.subheader(f"Recurring Cone -- {fmt(dca_data['contrib'], cs)}/mo")
        dca_fan = dca_data["fan"]
        df_d = pd.DataFrame(dca_fan)
        base_d = alt.Chart(df_d).encode(x=alt.X("years:Q", title="Years held"))
        layers_d = []
        for lo, hi, op in bands:
            layers_d.append(
                base_d.mark_area(opacity=op, color="#1565c0").encode(
                    y=alt.Y(f"{lo}:Q", title=f"Portfolio value ({cs})"), y2=f"{hi}:Q"))
        layers_d.append(base_d.mark_line(color="#0d47a1", strokeWidth=2.5).encode(y="p50:Q"))
        invested_df = pd.DataFrame({"years": dca_fan["years"], "invested": dca_fan["invested"]})
        invested_step = alt.Chart(invested_df).mark_line(
            color="gray", strokeDash=[5, 5], strokeWidth=1.5).encode(
            x="years:Q", y="invested:Q")
        st.altair_chart(
            alt.layer(*layers_d, invested_step).properties(height=350).interactive(),
            width="stretch")
        st.caption("Dashed line = total invested (rises with each monthly contribution).")

with st.expander("How to read this / limitations"):
    st.markdown(
        "- **P(profit)** = share of simulated futures ending above what you put in.\n"
        "- **VaR/CVaR** = the bad tail: where you end in the worst 5%, and its average.\n"
        "- **Drawdown** = worst peak-to-trough dip you'd sit through, even on good paths.\n"
        "- **Lump sum vs recurring** see identical market scenarios (same bootstrap draws); "
        "only the cash-flow timing differs. Lump sum typically has higher median (more time "
        "in market) but wider range; recurring has lower variance (less capital at risk early).\n"
        "- The threshold flag is a **mechanical comparison on a historical prior -- not advice "
        "and not a trading signal**. All figures are **nominal** (not inflation-adjusted)."
    )
