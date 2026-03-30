# ============================================================
#  backend.py  —  Layer 3: Backend Python ML / Analysis Engine
#  Computes all financial metrics, risk classification,
#  and generates investment advice
# ============================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


# ──────────────────────────────────────────────────────────────
#  Data classes for structured output
# ──────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    """All quantitative risk and performance metrics."""
    # Returns
    total_return_pct:   float = 0.0
    annualized_return:  float = 0.0
    # Risk
    volatility:         float = 0.0   # annualized std dev of returns (%)
    max_drawdown:       float = 0.0   # worst peak-to-trough drop (%)
    var_95:             float = 0.0   # Value-at-Risk at 95% confidence (%)
    # Market sensitivity
    beta:               float = 1.0   # correlation to S&P 500
    # Risk-adjusted performance
    sharpe_ratio:       float = 0.0   # excess return per unit of risk
    sortino_ratio:      float = 0.0   # downside-deviation adjusted return
    calmar_ratio:       float = 0.0   # return / max drawdown
    # Technical
    rsi:                float = 50.0  # current RSI value
    macd:               float = 0.0
    macd_signal:        float = 0.0
    # Moving average signals
    ma20:               float = 0.0
    ma50:               float = 0.0
    ma200:              float = 0.0
    current_price:      float = 0.0


@dataclass
class RiskScore:
    """Composite risk classification output."""
    level:       Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    score:       int  = 0          # raw score (0–16)
    confidence:  str  = "Medium"   # Low / Medium / High
    components:  dict = field(default_factory=dict)   # per-metric sub-scores


@dataclass
class InvestmentAdvice:
    """Structured investment recommendation."""
    action:         str = ""   # BUY / HOLD / SELL / AVOID
    time_horizon:   str = ""
    position_size:  str = ""
    summary:        str = ""
    pros:           list = field(default_factory=list)
    cons:           list = field(default_factory=list)
    stop_loss:      str = ""
    target_price:   str = ""


# ──────────────────────────────────────────────────────────────
#  Core calculation functions
# ──────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, market_returns: pd.Series) -> RiskMetrics:
    """
    Compute all risk and performance metrics from a prepared DataFrame
    (must include Daily_Return, RSI, MACD, Signal, MA20, MA50, MA200 columns).

    Args:
        df             : DataFrame with technical indicators already added
        market_returns : S&P 500 (SPY) daily returns Series

    Returns:
        RiskMetrics dataclass
    """
    m = RiskMetrics()
    returns = df["Daily_Return"].dropna()

    if returns.empty:
        return m

    # ── Current price and MAs
    m.current_price = round(float(df["Close"].iloc[-1]), 2)
    m.ma20          = round(float(df["MA20"].iloc[-1]),  2) if not pd.isna(df["MA20"].iloc[-1])  else 0.0
    m.ma50          = round(float(df["MA50"].iloc[-1]),  2) if not pd.isna(df["MA50"].iloc[-1])  else 0.0
    m.ma200         = round(float(df["MA200"].iloc[-1]), 2) if not pd.isna(df["MA200"].iloc[-1]) else 0.0
    m.rsi           = round(float(df["RSI"].iloc[-1]),   1) if not pd.isna(df["RSI"].iloc[-1])   else 50.0
    m.macd          = round(float(df["MACD"].iloc[-1]),  4) if not pd.isna(df["MACD"].iloc[-1])  else 0.0
    m.macd_signal   = round(float(df["Signal"].iloc[-1]),4) if not pd.isna(df["Signal"].iloc[-1])else 0.0

    # ── Total return
    start_price = float(df["Close"].iloc[0])
    end_price   = float(df["Close"].iloc[-1])
    m.total_return_pct = round(((end_price - start_price) / start_price) * 100, 2)

    # ── Annualized return (CAGR)
    n_years = len(df) / 252
    if n_years > 0 and start_price > 0:
        m.annualized_return = round(((end_price / start_price) ** (1 / n_years) - 1) * 100, 2)

    # ── Volatility (annualized std dev)
    m.volatility = round(float(returns.std() * np.sqrt(252) * 100), 2)

    # ── Max Drawdown
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    m.max_drawdown = round(float(drawdown.min() * 100), 2)

    # ── Value at Risk (95%)
    m.var_95 = round(float(np.percentile(returns, 5) * 100), 2)

    # ── Beta (vs S&P 500)
    m.beta = _compute_beta(returns, market_returns)

    # ── Sharpe Ratio (risk-free rate = 5% annual)
    rf_daily = 0.05 / 252
    excess   = returns - rf_daily
    if returns.std() > 0:
        m.sharpe_ratio = round(float((excess.mean() / returns.std()) * np.sqrt(252)), 2)

    # ── Sortino Ratio (only downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        m.sortino_ratio = round(float((excess.mean() / downside.std()) * np.sqrt(252)), 2)

    # ── Calmar Ratio
    if m.max_drawdown != 0:
        m.calmar_ratio = round(abs(m.annualized_return / m.max_drawdown), 2)

    return m


def _compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate Beta coefficient vs market (S&P 500)."""
    try:
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned) < 30:
            return 1.0
        aligned.columns = ["stock", "market"]
        cov = aligned.cov().iloc[0, 1]
        var = aligned["market"].var()
        return round(float(cov / var), 2) if var != 0 else 1.0
    except Exception:
        return 1.0


# ──────────────────────────────────────────────────────────────
#  Risk Classification Model
# ──────────────────────────────────────────────────────────────

def classify_risk(m: RiskMetrics) -> RiskScore:
    """
    Rule-based scoring model that maps financial metrics to a
    composite risk level.

    Scoring grid (each metric contributes 0–4 points):
        Metric          | Low (0–1) | Medium (2) | High (3–4)
        ─────────────────────────────────────────────────────
        Volatility      | <15%      | 15–30%     | >30%
        Max Drawdown    | >-10%     | -10 to -25 | <-25%
        Sharpe Ratio    | >1.5      | 0.5–1.5    | <0.5
        Beta            | <0.8      | 0.8–1.2    | >1.2
        VaR 95%         | >-1.5%    | -1.5 to -3 | <-3%

    Total score → 0–5: LOW   |  6–10: MEDIUM   |  11–20: HIGH
    """
    components = {}
    score = 0

    # Volatility
    if m.volatility > 40:
        v = 4
    elif m.volatility > 30:
        v = 3
    elif m.volatility > 20:
        v = 2
    elif m.volatility > 12:
        v = 1
    else:
        v = 0
    components["volatility"] = v
    score += v

    # Max Drawdown
    if m.max_drawdown < -35:
        d = 4
    elif m.max_drawdown < -25:
        d = 3
    elif m.max_drawdown < -15:
        d = 2
    elif m.max_drawdown < -8:
        d = 1
    else:
        d = 0
    components["max_drawdown"] = d
    score += d

    # Sharpe Ratio
    if m.sharpe_ratio >= 2.0:
        s = 0
    elif m.sharpe_ratio >= 1.0:
        s = 1
    elif m.sharpe_ratio >= 0.5:
        s = 2
    elif m.sharpe_ratio >= 0:
        s = 3
    else:
        s = 4
    components["sharpe_ratio"] = s
    score += s

    # Beta
    if m.beta > 2.0:
        b = 4
    elif m.beta > 1.5:
        b = 3
    elif m.beta > 1.0:
        b = 2
    elif m.beta > 0.7:
        b = 1
    else:
        b = 0
    components["beta"] = b
    score += b

    # Value at Risk
    if m.var_95 < -4:
        var_s = 4
    elif m.var_95 < -2.5:
        var_s = 3
    elif m.var_95 < -1.5:
        var_s = 2
    elif m.var_95 < -0.8:
        var_s = 1
    else:
        var_s = 0
    components["var_95"] = var_s
    score += var_s

    # Classify
    if score <= 5:
        level = "LOW"
    elif score <= 11:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # Confidence based on consistency of sub-scores
    values = list(components.values())
    spread = max(values) - min(values)
    confidence = "High" if spread <= 1 else ("Medium" if spread <= 2 else "Low")

    return RiskScore(
        level=level,
        score=score,
        confidence=confidence,
        components=components,
    )


# ──────────────────────────────────────────────────────────────
#  Investment Advisor
# ──────────────────────────────────────────────────────────────

def generate_advice(m: RiskMetrics, risk: RiskScore, summary: dict) -> InvestmentAdvice:
    """
    Generate structured investment advice based on risk level,
    technical signals, and fundamental data.
    """
    advice = InvestmentAdvice()
    price  = m.current_price
    rsi    = m.rsi

    # ── Determine action
    bullish_signals = 0
    bearish_signals = 0

    if price > m.ma50:  bullish_signals += 1
    else:               bearish_signals += 1

    if price > m.ma200: bullish_signals += 1
    else:               bearish_signals += 1

    if m.macd > m.macd_signal:  bullish_signals += 1
    else:                        bearish_signals += 1

    if rsi < 30:   bullish_signals += 1   # oversold → potential reversal
    elif rsi > 70: bearish_signals += 1   # overbought → potential pullback

    if risk.level == "LOW":
        if bullish_signals >= 3:
            advice.action = "BUY"
        elif bearish_signals >= 3:
            advice.action = "HOLD"
        else:
            advice.action = "BUY / HOLD"
        advice.time_horizon  = "Medium to Long term (1–5 years)"
        advice.position_size = "Up to 10–15% of portfolio"

    elif risk.level == "MEDIUM":
        if bullish_signals >= 3:
            advice.action = "BUY (selective)"
        elif bearish_signals >= 3:
            advice.action = "HOLD / REDUCE"
        else:
            advice.action = "HOLD"
        advice.time_horizon  = "Medium term (1–3 years)"
        advice.position_size = "5–10% of portfolio"

    else:  # HIGH
        if bullish_signals >= 3 and rsi < 40:
            advice.action = "SPECULATIVE BUY"
        else:
            advice.action = "AVOID / SELL"
        advice.time_horizon  = "Short term only (< 1 year)"
        advice.position_size = "Max 3–5% of portfolio"

    # ── Stop loss & target price
    advice.stop_loss    = f"${round(price * 0.92, 2):,.2f}  (−8% from current)"
    advice.target_price = f"${round(price * 1.15, 2):,.2f}  (+15% from current)"

    # ── Pros
    if m.sharpe_ratio > 1:
        advice.pros.append(f"Strong risk-adjusted returns (Sharpe = {m.sharpe_ratio})")
    if m.total_return_pct > 10:
        advice.pros.append(f"Solid historical performance (+{m.total_return_pct}% over period)")
    if price > m.ma50:
        advice.pros.append("Price trading above 50-day moving average (bullish trend)")
    if rsi < 45:
        advice.pros.append(f"RSI at {rsi} — not overbought, room to grow")
    if m.beta < 1:
        advice.pros.append(f"Low Beta ({m.beta}) — moves less than market (defensive)")
    if m.macd > m.macd_signal:
        advice.pros.append("MACD above signal line — bullish momentum")
    if not advice.pros:
        advice.pros.append("Monitor for improved entry conditions")

    # ── Cons
    if m.volatility > 30:
        advice.cons.append(f"High volatility ({m.volatility}%) — large price swings")
    if m.max_drawdown < -20:
        advice.cons.append(f"Significant max drawdown ({m.max_drawdown}%) — historical large losses")
    if rsi > 65:
        advice.cons.append(f"RSI at {rsi} — approaching overbought territory")
    if m.beta > 1.3:
        advice.cons.append(f"High Beta ({m.beta}) — amplifies market downturns")
    if m.sharpe_ratio < 0.5:
        advice.cons.append(f"Poor risk-adjusted return (Sharpe = {m.sharpe_ratio})")
    if price < m.ma200:
        advice.cons.append("Price below 200-day MA — long-term downtrend signal")
    if not advice.cons:
        advice.cons.append("No major red flags detected at this time")

    # ── Summary text
    level_desc = {
        "LOW":    "a relatively stable, low-volatility asset",
        "MEDIUM": "a moderately risky asset with balanced risk-reward",
        "HIGH":   "a high-risk, high-volatility asset requiring caution",
    }
    advice.summary = (
        f"{summary.get('company_name', 'This stock')} is classified as {level_desc[risk.level]}. "
        f"The composite risk score is {risk.score}/20 with {risk.confidence.lower()} confidence. "
        f"Based on technical indicators and risk metrics, the recommended action is: {advice.action}."
    )

    return advice
