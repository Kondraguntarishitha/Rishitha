# ============================================================
#  app.py  —  Layer 2: Frontend (Streamlit UI)
#  Complete user-facing dashboard for stock risk analysis
#
#  Run:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our own layers
from stock_data import fetch_stock_data, fetch_market_data, add_technical_indicators, get_summary_stats
from backend    import compute_metrics, classify_risk, generate_advice


# ─── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense — AI Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── App shell ── */
.stApp { background: #070a10; color: #c9d1e0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c1018;
    border-right: 1px solid #141c2a;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #101520 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 8px !important;
    color: #c9d1e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a56e8, #0e3ab0);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1a4fcc);
    transform: translateY(-1px);
    color: white;
}

/* ── Cards ── */
.card {
    background: #0c1018;
    border: 1px solid #141c2a;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #3d5070;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: #e8edf5;
    line-height: 1.2;
}
.card-sub {
    font-size: 0.78rem;
    color: #3d5070;
    margin-top: 0.3rem;
}

/* ── Risk badge ── */
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.45rem 1.1rem;
    border-radius: 50px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.risk-LOW    { background:#071a10; color:#22d47e; border:1px solid #0d4425; }
.risk-MEDIUM { background:#1a1200; color:#f5a623; border:1px solid #4a3300; }
.risk-HIGH   { background:#1a0707; color:#f05353; border:1px solid #4a1414; }

/* ── Action badge ── */
.action-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.action-BUY        { background:#0d2e1a; color:#22d47e; border:1px solid #1a5e35; }
.action-HOLD       { background:#1a1a00; color:#f5d423; border:1px solid #4a4400; }
.action-SELL       { background:#1a0a0a; color:#f05353; border:1px solid #4a1515; }
.action-AVOID      { background:#1a0a0a; color:#f05353; border:1px solid #4a1515; }
.action-SPECULATIVE{ background:#1a0d00; color:#f5832a; border:1px solid #4a2a00; }

/* ── Section heading ── */
.section-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #1a56e8;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 2rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-head::after {
    content:'';
    flex:1;
    height:1px;
    background:#141c2a;
}

/* ── Pros / Cons lists ── */
.pro-item  { color:#22d47e; font-size:0.88rem; padding:0.3rem 0; }
.con-item  { color:#f05353; font-size:0.88rem; padding:0.3rem 0; }
.list-icon { margin-right:6px; }

/* ── Score bar ── */
.score-track {
    background:#141c2a;
    border-radius:50px;
    height:8px;
    overflow:hidden;
    margin-top:6px;
}
.score-fill {
    height:100%;
    border-radius:50px;
    transition: width 0.6s ease;
}

/* ── Info box ── */
.info-box {
    background:#0c1018;
    border-left:3px solid #1a56e8;
    border-radius:0 10px 10px 0;
    padding:1rem 1.2rem;
    font-size:0.88rem;
    color:#8094b0;
    line-height:1.7;
}

/* ── Metric change ── */
.up   { color:#22d47e; }
.down { color:#f05353; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display:none; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cached data fetching ────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(ticker: str, period: str):
    df_raw, info        = fetch_stock_data(ticker, period)
    market_returns      = fetch_market_data(period)
    df                  = add_technical_indicators(df_raw)
    summary             = get_summary_stats(df, info)
    metrics             = compute_metrics(df, market_returns)
    risk                = classify_risk(metrics)
    advice              = generate_advice(metrics, risk, summary)
    return df, summary, metrics, risk, advice


# ─── Chart helpers ───────────────────────────────────────────────────────────
CHART_BG   = "#070a10"
CHART_GRID = "#111826"
CHART_TEXT = "#3d5070"
FONT       = "JetBrains Mono"

def _base_layout(title="", height=360):
    return dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family=FONT, color=CHART_TEXT, size=11),
        title=dict(text=title, font=dict(color="#8094b0", size=13), x=0.01),
        margin=dict(l=12, r=12, t=44, b=12),
        height=height,
        xaxis=dict(showgrid=False, color=CHART_TEXT, zeroline=False,
                   showline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=CHART_GRID, color=CHART_TEXT,
                   zeroline=False, showline=False, tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=10, color="#5a7090")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0c1422", font_color="#c9d1e0",
                        font_family=FONT, font_size=11, bordercolor="#1a2235"),
    )


def chart_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Price + MA + Bollinger Band chart."""
    fig = go.Figure()

    # Bollinger Band fill
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["BB_upper"]) + list(df["BB_lower"][::-1]),
        fill="toself", fillcolor="rgba(26,86,232,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="Bollinger Band",
        hoverinfo="skip", showlegend=True,
    ))

    # Price area
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#1a56e8", width=1.8),
        fill="tozeroy", fillcolor="rgba(26,86,232,0.04)",
    ))

    # MAs
    for col, color, label in [
        ("MA20",  "#f5a623", "MA20"),
        ("MA50",  "#f05353", "MA50"),
        ("MA200", "#22d47e", "MA200"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=label,
            line=dict(color=color, width=1, dash="dot"),
        ))

    fig.update_layout(**_base_layout(f"{ticker} — Price & Moving Averages", 380))
    return fig


def chart_volume(df: pd.DataFrame) -> go.Figure:
    """Volume bar chart."""
    colors = ["#22d47e" if c >= o else "#f05353"
              for c, o in zip(df["Close"], df["Open"])]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="Volume",
    ))
    fig.update_layout(**_base_layout("Volume", 200))
    return fig


def chart_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    fig = go.Figure()

    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(240,83,83,0.08)",
                  line_width=0, annotation_text="Overbought",
                  annotation_position="top left",
                  annotation_font=dict(color="#f05353", size=9))
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,212,126,0.08)",
                  line_width=0, annotation_text="Oversold",
                  annotation_position="bottom left",
                  annotation_font=dict(color="#22d47e", size=9))
    fig.add_hline(y=70, line_dash="dot", line_color="#f05353", line_width=0.6)
    fig.add_hline(y=30, line_dash="dot", line_color="#22d47e", line_width=0.6)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        line=dict(color="#8b5cf6", width=1.6), name="RSI (14)",
    ))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(**_base_layout("RSI (14-day)", 220))
    return fig


def chart_macd(df: pd.DataFrame) -> go.Figure:
    """MACD + Signal + Histogram."""
    hist   = df["MACD"] - df["Signal"]
    colors = ["#22d47e" if v >= 0 else "#f05353" for v in hist]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=colors,
                         name="Histogram", opacity=0.6))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                             line=dict(color="#1a56e8", width=1.4), name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"],
                             line=dict(color="#f5a623", width=1.2, dash="dot"),
                             name="Signal"))
    fig.update_layout(**_base_layout("MACD", 230))
    return fig


def chart_returns_dist(df: pd.DataFrame) -> go.Figure:
    """Daily returns histogram with normal fit."""
    rets = df["Daily_Return"].dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=60,
        marker_color="#1a56e8", opacity=0.75, name="Daily Returns",
    ))
    fig.update_layout(**_base_layout("Daily Returns Distribution (%)", 260))
    fig.update_xaxes(title_text="Return (%)", title_font=dict(size=10))
    return fig


def chart_drawdown(df: pd.DataFrame) -> go.Figure:
    """Underwater equity (drawdown) chart."""
    returns    = df["Daily_Return"].dropna()
    cum        = (1 + returns).cumprod()
    roll_max   = cum.cummax()
    drawdown   = ((cum - roll_max) / roll_max) * 100

    fig = go.Figure(go.Scatter(
        x=drawdown.index, y=drawdown,
        line=dict(color="#f05353", width=1.2),
        fill="tozeroy", fillcolor="rgba(240,83,83,0.08)",
        name="Drawdown %",
    ))
    fig.update_layout(**_base_layout("Drawdown (%)", 230))
    return fig


def chart_risk_radar(risk_score: dict) -> go.Figure:
    """Spider/radar chart of per-metric risk sub-scores."""
    cats   = ["Volatility", "Max Drawdown", "Sharpe Ratio", "Beta", "VaR 95%"]
    keys   = ["volatility", "max_drawdown", "sharpe_ratio", "beta", "var_95"]
    values = [risk_score.get(k, 0) for k in keys]
    values_closed = values + [values[0]]
    cats_closed   = cats   + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(26,86,232,0.12)",
        line=dict(color="#1a56e8", width=1.5),
        name="Risk Score",
    ))
    fig.update_polars(
        bgcolor=CHART_BG,
        radialaxis=dict(
            visible=True, range=[0, 4],
            gridcolor=CHART_GRID, color=CHART_TEXT,
            tickfont=dict(size=9),
        ),
        angularaxis=dict(color="#5a7090", gridcolor=CHART_GRID, tickfont=dict(size=10)),
    )
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        font=dict(family=FONT, color=CHART_TEXT, size=10),
        margin=dict(l=30, r=30, t=44, b=30),
        height=290,
        title=dict(text="Risk Components", font=dict(color="#8094b0", size=13), x=0.01),
        showlegend=False,
    )
    return fig


# ─── Small UI helpers ────────────────────────────────────────────────────────

def _card(label, value, sub="", color="#e8edf5"):
    return f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="color:{color}">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>"""

def _section(title: str):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)

def _fmt_mcap(val):
    if val is None: return "—"
    if val >= 1e12: return f"${val/1e12:.2f}T"
    if val >= 1e9:  return f"${val/1e9:.2f}B"
    if val >= 1e6:  return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"

def _score_bar(score: int, max_score: int = 20):
    pct   = int(score / max_score * 100)
    color = "#22d47e" if pct < 35 else ("#f5a623" if pct < 60 else "#f05353")
    return f"""
    <div class="score-track">
      <div class="score-fill" style="width:{pct}%;background:{color}"></div>
    </div>"""


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-size:1.1rem;font-weight:800;color:#e8edf5;letter-spacing:-0.5px">
        📊 StockSense
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                  color:#3d5070;letter-spacing:2px;margin-top:3px">
        AI RISK ANALYZER
      </div>
    </div>
    """, unsafe_allow_html=True)

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="AAPL · TSLA · TCS.NS",
        help="NSE stocks: add .NS   |   BSE: add .BO",
    )

    period_options = {
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year":   "1y",
        "2 Years":  "2y",
        "5 Years":  "5y",
    }
    selected_period_label = st.selectbox(
        "Analysis Period", list(period_options.keys()), index=2
    )
    selected_period = period_options[selected_period_label]

    analyze = st.button("🔍  Analyze Stock")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#3d5070;line-height:1.8">
      <b style="color:#5a7090">Popular tickers</b><br>
      AAPL · MSFT · TSLA · GOOGL<br>
      AMZN · NVDA · META · NFLX<br><br>
      <b style="color:#5a7090">Indian stocks</b><br>
      TCS.NS · INFY.NS · RELIANCE.NS<br>
      WIPRO.NS · HDFCBANK.NS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;font-family:'JetBrains Mono',monospace;
                font-size:0.6rem;color:#1a2235;line-height:2">
      DATA: Yahoo Finance<br>
      RISK MODEL: Rule-based scoring<br>
      NOT financial advice
    </div>
    """, unsafe_allow_html=True)


# ─── Main page ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 1.5rem 0; border-bottom:1px solid #141c2a; margin-bottom:1.5rem">
  <h1 style="font-size:1.8rem;font-weight:800;color:#e8edf5;margin:0;letter-spacing:-1px">
    AI-Based Stock Risk Analysis
  </h1>
  <p style="color:#3d5070;font-size:0.85rem;margin:0.3rem 0 0 0;
            font-family:'JetBrains Mono',monospace;letter-spacing:1px">
    INVESTMENT ADVISORY PLATFORM
  </p>
</div>
""", unsafe_allow_html=True)

# ── Welcome state
if not analyze and "last_ticker" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem">
      <div style="font-size:3.5rem;margin-bottom:1rem">📊</div>
      <h3 style="color:#3d5070;font-weight:500;font-size:1.1rem">
        Enter a stock ticker in the sidebar and click Analyze
      </h3>
      <p style="color:#1e2a3a;font-size:0.85rem;margin-top:0.5rem">
        Supports NYSE · NASDAQ · NSE · BSE
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run analysis
ticker = ticker_input.strip().upper()
if analyze:
    st.session_state["last_ticker"] = ticker
    st.session_state["last_period"] = selected_period
else:
    ticker          = st.session_state.get("last_ticker", ticker)
    selected_period = st.session_state.get("last_period", selected_period)

with st.spinner(f"Fetching & analyzing  {ticker}  …"):
    try:
        df, summary, metrics, risk, advice = load_all_data(ticker, selected_period)
    except ValueError as e:
        st.error(f"❌  {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌  Unexpected error: {e}")
        st.stop()


# ════════════════════════════════════════════════════════════════
#  SECTION 1 — Company header
# ════════════════════════════════════════════════════════════════
col_info, col_risk = st.columns([3, 1])

with col_info:
    change_color = "#22d47e" if summary["day_change"] >= 0 else "#f05353"
    change_sign  = "+" if summary["day_change"] >= 0 else ""
    st.markdown(f"""
    <div style="margin-bottom:0.5rem">
      <span style="font-size:1.5rem;font-weight:800;color:#e8edf5">
        {summary["company_name"]}
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                   color:#3d5070;margin-left:10px;letter-spacing:1.5px">
        {summary["ticker"]} · {summary["exchange"]}
      </span>
    </div>
    <div style="font-size:0.82rem;color:#3d5070;margin-bottom:0.8rem">
      {summary["sector"]} &nbsp;·&nbsp; {summary["industry"]}
    </div>
    <div>
      <span style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                   font-weight:600;color:#e8edf5">
        {summary["currency"]} {summary["current_price"]:,.2f}
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;
                   color:{change_color};margin-left:12px">
        {change_sign}{summary["day_change"]:,.2f}
        ({change_sign}{summary["day_change_pct"]:.2f}%)
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_risk:
    st.markdown(f"""
    <div style="text-align:right;padding-top:0.5rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                  color:#3d5070;letter-spacing:2px;margin-bottom:6px">
        RISK LEVEL
      </div>
      <span class="risk-pill risk-{risk.level}">
        ● &nbsp; {risk.level}
      </span>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                  color:#3d5070;margin-top:8px">
        Score: {risk.score}/20  ·  {risk.confidence} confidence
      </div>
      {_score_bar(risk.score)}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='border-bottom:1px solid #141c2a;margin:1rem 0'></div>",
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 2 — Key metrics row
# ════════════════════════════════════════════════════════════════
_section("Key Metrics")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def _pct_color(val):
    return "#22d47e" if val >= 0 else "#f05353"

cols_data = [
    (c1, "Volatility",      f"{metrics.volatility}%",        "Ann. std dev"),
    (c2, "Sharpe Ratio",    str(metrics.sharpe_ratio),       ">1.0 is good"),
    (c3, "Beta",            str(metrics.beta),               "vs S&P 500"),
    (c4, "Max Drawdown",    f"{metrics.max_drawdown}%",      "Worst loss"),
    (c5, "VaR 95%",         f"{metrics.var_95}%",            "Daily worst case"),
    (c6, "Sortino",         str(metrics.sortino_ratio),      "Downside risk adj."),
    (c7, "1Y Return",       f"{metrics.total_return_pct}%",  "Total period return"),
]

for col, label, val, sub in cols_data:
    color = "#e8edf5"
    if "%" in val:
        try:
            n = float(val.replace("%", ""))
            if label in ("Volatility", "Max Drawdown", "VaR 95%"):
                color = "#f05353" if n < -10 or n > 30 else "#f5a623"
            else:
                color = _pct_color(n)
        except Exception:
            pass
    col.markdown(_card(label, val, sub, color), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 2.5 — Gains & Performance
# ════════════════════════════════════════════════════════════════
_section("Gains & Performance")

# Calculate gains metrics
period_return = metrics.total_return_pct
period_return_color = "#22d47e" if period_return >= 0 else "#f05353"
period_return_sign = "+" if period_return >= 0 else ""

# 52-week high/low gains
high_52w = summary.get("52w_high", 0)
low_52w = summary.get("52w_low", 0)
curr_price = summary.get("current_price", 0)
if high_52w and low_52w and curr_price:
    range_gain = ((curr_price - low_52w) / low_52w) * 100 if low_52w > 0 else 0
    from_high_loss = ((curr_price - high_52w) / high_52w) * 100 if high_52w > 0 else 0
else:
    range_gain = 0
    from_high_loss = 0

g1, g2, g3, g4 = st.columns(4)

g1.markdown(_card(
    "Period Return",
    f"{period_return_sign}{period_return}%",
    f"Total {selected_period_label} gain",
    period_return_color
), unsafe_allow_html=True)

g2.markdown(_card(
    "From 52W Low",
    f"+{range_gain:.2f}%" if range_gain >= 0 else f"{range_gain:.2f}%",
    "Gain from yearly low",
    "#22d47e" if range_gain >= 0 else "#f05353"
), unsafe_allow_html=True)

g3.markdown(_card(
    "vs 52W High",
    f"{from_high_loss:.2f}%",
    "vs yearly high",
    "#22d47e" if from_high_loss >= 0 else "#f05353"
), unsafe_allow_html=True)

g4.markdown(_card(
    "Ann. Gain",
    f"{period_return_sign}{metrics.annualized_return}%",
    "Annualized CAGR",
    period_return_color
), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 2.6 — Price Increase/Decrease Trend
# ════════════════════════════════════════════════════════════════
_section("Price Trend Analysis")

# Calculate price trends
if len(df) >= 2:
    current_price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    open_price = float(df["Open"].iloc[-1])
    high_price = float(df["High"].iloc[-1])
    low_price = float(df["Low"].iloc[-1])
    
    # Today's change
    today_change = current_price - open_price
    today_change_pct = (today_change / open_price * 100) if open_price > 0 else 0
    
    # Week trend (if enough data)
    week_ago = float(df["Close"].iloc[-min(5, len(df)-1)])
    week_change = current_price - week_ago
    week_change_pct = (week_change / week_ago * 100) if week_ago > 0 else 0
    
    # Month trend (if enough data)
    month_ago = float(df["Close"].iloc[-min(21, len(df)-1)])
    month_change = current_price - month_ago
    month_change_pct = (month_change / month_ago * 100) if month_ago > 0 else 0
    
    # Determine trend direction
    today_trend = "📈 UP" if today_change >= 0 else "📉 DOWN"
    week_trend = "📈 UP" if week_change >= 0 else "📉 DOWN"
    month_trend = "📈 UP" if month_change >= 0 else "📉 DOWN"
    
    today_color = "#22d47e" if today_change >= 0 else "#f05353"
    week_color = "#22d47e" if week_change >= 0 else "#f05353"
    month_color = "#22d47e" if month_change >= 0 else "#f05353"
    
    t1, t2, t3, t4 = st.columns(4)
    
    t1.markdown(_card(
        "Today Change",
        f"{today_trend}\n{abs(today_change_pct):.2f}%",
        f"${abs(today_change):.2f}",
        today_color
    ), unsafe_allow_html=True)
    
    t2.markdown(_card(
        "This Week",
        f"{week_trend}\n{abs(week_change_pct):.2f}%",
        f"${abs(week_change):.2f}",
        week_color
    ), unsafe_allow_html=True)
    
    t3.markdown(_card(
        "This Month",
        f"{month_trend}\n{abs(month_change_pct):.2f}%",
        f"${abs(month_change):.2f}",
        month_color
    ), unsafe_allow_html=True)
    
    # Current price details
    t4.markdown(_card(
        "Daily Range",
        f"${high_price:.2f}",
        f"H: ${high_price:.2f} | L: ${low_price:.2f}",
        "#8b5cf6"
    ), unsafe_allow_html=True)
    
    # Price momentum bar
    st.markdown(f"""
    <div style="margin-top:1.5rem;padding:1.5rem;background:#0c1018;border:1px solid #141c2a;border-radius:14px">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#3d5070;
                    text-transform:uppercase;letter-spacing:2px;margin-bottom:1rem">
            📊 Price Momentum
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">
            <div>
                <div style="font-size:0.75rem;color:#5a7090;margin-bottom:0.5rem">Today's Movement</div>
                <div style="font-size:1.2rem;font-weight:600;color:{today_color}">{today_trend} {abs(today_change_pct):.2f}%</div>
                <div style="font-size:0.7rem;color:#3d5070;margin-top:0.3rem">Open: ${open_price:.2f} → Close: ${current_price:.2f}</div>
            </div>
            <div>
                <div style="font-size:0.75rem;color:#5a7090;margin-bottom:0.5rem">Period Performance</div>
                <div style="font-size:1.2rem;font-weight:600;color:{period_return_color}">{period_return_sign}{period_return}% {period_return_sign if period_return >= 0 else ''}({selected_period_label})</div>
                <div style="font-size:0.7rem;color:#3d5070;margin-top:0.3rem">Total period gain/loss since start</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 3 — Charts row 1: Price + Volume
# ════════════════════════════════════════════════════════════════
_section("Price Chart")
st.plotly_chart(chart_price(df, ticker), use_container_width=True)
st.plotly_chart(chart_volume(df), use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 4 — Charts row 2: RSI · MACD · Returns · Drawdown
# ════════════════════════════════════════════════════════════════
_section("Technical Indicators")
tc1, tc2 = st.columns(2)
with tc1:
    st.plotly_chart(chart_rsi(df),  use_container_width=True)
    st.plotly_chart(chart_macd(df), use_container_width=True)
with tc2:
    st.plotly_chart(chart_returns_dist(df), use_container_width=True)
    st.plotly_chart(chart_drawdown(df),     use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 5 — Risk breakdown + Radar
# ════════════════════════════════════════════════════════════════
_section("Risk Breakdown")
rb1, rb2 = st.columns([1, 1])

with rb1:
    st.markdown("**Sub-scores per metric** (0 = safe, 4 = very risky)")
    for key, label in [
        ("volatility",   "Volatility"),
        ("max_drawdown", "Max Drawdown"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("beta",         "Beta"),
        ("var_95",       "VaR 95%"),
    ]:
        val  = risk.components.get(key, 0)
        pct  = int(val / 4 * 100)
        col  = "#22d47e" if val <= 1 else ("#f5a623" if val <= 2 else "#f05353")
        st.markdown(f"""
        <div style="margin-bottom:0.7rem">
          <div style="display:flex;justify-content:space-between;
                      font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                      color:#5a7090;margin-bottom:3px">
            <span>{label}</span>
            <span style="color:{col}">{val}/4</span>
          </div>
          <div class="score-track">
            <div class="score-fill" style="width:{pct}%;background:{col}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with rb2:
    st.plotly_chart(chart_risk_radar(risk.components), use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 6 — Technical signals
# ════════════════════════════════════════════════════════════════
_section("Technical Signals")

s1, s2, s3, s4, s5 = st.columns(5)

rsi_signal  = "Oversold 🟢"   if metrics.rsi < 30 else ("Overbought 🔴" if metrics.rsi > 70 else "Neutral 🟡")
ma_signal   = "Bullish 🟢"    if metrics.current_price > metrics.ma50  else "Bearish 🔴"
ma200_sig   = "Above MA200 🟢" if metrics.current_price > metrics.ma200 else "Below MA200 🔴"
macd_signal = "Bullish 🟢"    if metrics.macd > metrics.macd_signal    else "Bearish 🔴"
ann_ret_col = "#22d47e" if metrics.annualized_return >= 0 else "#f05353"

s1.markdown(_card("RSI (14-day)",       f"{metrics.rsi}",               rsi_signal),  unsafe_allow_html=True)
s2.markdown(_card("Price vs MA50",      f"${metrics.ma50:,.2f}",        ma_signal),   unsafe_allow_html=True)
s3.markdown(_card("Price vs MA200",     f"${metrics.ma200:,.2f}",       ma200_sig),   unsafe_allow_html=True)
s4.markdown(_card("MACD Signal",        f"{metrics.macd:.3f}",          macd_signal), unsafe_allow_html=True)
s5.markdown(_card("Ann. Return",        f"{metrics.annualized_return}%","CAGR", ann_ret_col), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 7 — Fundamentals
# ════════════════════════════════════════════════════════════════
_section("Fundamentals")
f1, f2, f3, f4 = st.columns(4)
f1.markdown(_card("Market Cap",    _fmt_mcap(summary["market_cap"]),   "Total market value"), unsafe_allow_html=True)
f2.markdown(_card("P/E Ratio",     str(round(summary["pe_ratio"], 2)) if summary["pe_ratio"] else "—", "Trailing P/E"), unsafe_allow_html=True)
f3.markdown(_card("52W High",      f"${summary['52w_high']:,.2f}" if summary["52w_high"] else "—", "52-week high"),   unsafe_allow_html=True)
f4.markdown(_card("52W Low",       f"${summary['52w_low']:,.2f}"  if summary["52w_low"]  else "—", "52-week low"),    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SECTION 8 — Investment advice
# ════════════════════════════════════════════════════════════════
_section("Investment Advisory")

action_key = advice.action.split()[0]
action_css = f"action-{action_key}" if action_key in ("BUY","HOLD","SELL","AVOID","SPECULATIVE") else "action-HOLD"

st.markdown(f"""
<div class="card" style="border-color:#1a2a40">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;flex-wrap:wrap">
    <span style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;
                 color:#3d5070;letter-spacing:2px">RECOMMENDATION</span>
    <span class="action-pill {action_css}">{advice.action}</span>
    <span class="risk-pill risk-{risk.level}" style="font-size:0.65rem;padding:0.3rem 0.8rem">
      {risk.level} RISK
    </span>
  </div>
  <div class="info-box" style="margin-bottom:1rem">{advice.summary}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem">
    <div>
      <div class="card-label">Time Horizon</div>
      <div style="font-size:0.88rem;color:#8094b0">{advice.time_horizon}</div>
    </div>
    <div>
      <div class="card-label">Suggested Position</div>
      <div style="font-size:0.88rem;color:#8094b0">{advice.position_size}</div>
    </div>
    <div>
      <div class="card-label">Stop Loss / Target</div>
      <div style="font-size:0.82rem;color:#f05353">{advice.stop_loss}</div>
      <div style="font-size:0.82rem;color:#22d47e">{advice.target_price}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Pros & Cons
pros_col, cons_col = st.columns(2)
with pros_col:
    st.markdown('<div class="card-label" style="margin-bottom:0.5rem">✅ Strengths</div>', unsafe_allow_html=True)
    for p in advice.pros:
        st.markdown(f'<div class="pro-item">✦ {p}</div>', unsafe_allow_html=True)

with cons_col:
    st.markdown('<div class="card-label" style="margin-bottom:0.5rem">⚠️ Risks</div>', unsafe_allow_html=True)
    for c in advice.cons:
        st.markdown(f'<div class="con-item">✦ {c}</div>', unsafe_allow_html=True)


# ── Footer disclaimer
st.markdown("""
<div style="margin-top:3rem;padding:1rem;border-top:1px solid #141c2a;
            text-align:center;font-family:'JetBrains Mono',monospace;
            font-size:0.62rem;color:#1e2a3a;letter-spacing:1px;line-height:2">
  ⚠ THIS TOOL IS FOR EDUCATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.<br>
  Always consult a qualified financial advisor before making investment decisions.<br>
  DATA SOURCE: Yahoo Finance via yfinance
</div>
""", unsafe_allow_html=True)
