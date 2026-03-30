# ============================================================
#  stock_data.py  —  Layer 4: Stock Data API (Yahoo Finance)
#  Fetches and prepares all raw stock data from yfinance
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def fetch_stock_data(ticker: str, period: str = "1y") -> tuple[pd.DataFrame, dict]:
    """
    Fetch historical OHLCV data + company info from Yahoo Finance.

    Args:
        ticker : Stock symbol  e.g. "AAPL", "TCS.NS", "RELIANCE.NS"
        period : History length e.g. "6mo", "1y", "2y", "5y"

    Returns:
        df   : DataFrame with Open/High/Low/Close/Volume columns
        info : Dict of company metadata (name, sector, market cap …)
    """
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. "
                         "Check the symbol and try again.")

    # Clean column names
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    info = stock.info  # company metadata dict
    return df, info


def fetch_market_data(period: str = "1y") -> pd.Series:
    """
    Fetch S&P 500 (SPY) closes — used to compute Beta.
    Returns a Series of daily returns.
    """
    spy = yf.Ticker("SPY").history(period=period)
    if spy.empty:
        return pd.Series(dtype=float)
    return spy["Close"].pct_change().dropna()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to the OHLCV dataframe.
    Columns added:
        MA20, MA50, MA200   — Simple Moving Averages
        EMA12, EMA26        — Exponential Moving Averages (for MACD)
        MACD, Signal        — MACD line and signal line
        BB_upper, BB_lower  — Bollinger Bands (20-day, 2σ)
        RSI                 — Relative Strength Index (14-day)
        Daily_Return        — Day-over-day percentage change
    """
    df = df.copy()

    # ── Moving averages
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # ── MACD
    df["EMA12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]  = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands
    rolling_std      = df["Close"].rolling(20).std()
    df["BB_upper"]   = df["MA20"] + 2 * rolling_std
    df["BB_lower"]   = df["MA20"] - 2 * rolling_std

    # ── RSI
    delta       = df["Close"].diff()
    gain        = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss        = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs          = gain / loss
    df["RSI"]   = 100 - (100 / (1 + rs))

    # ── Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    return df


def get_summary_stats(df: pd.DataFrame, info: dict) -> dict:
    """
    Build a flat summary dictionary for quick display in the UI.
    """
    latest = df["Close"].iloc[-1]
    prev   = df["Close"].iloc[-2] if len(df) > 1 else latest
    change = latest - prev
    change_pct = (change / prev) * 100

    return {
        "ticker":       info.get("symbol", "—"),
        "company_name": info.get("longName", "—"),
        "sector":       info.get("sector", "—"),
        "industry":     info.get("industry", "—"),
        "exchange":     info.get("exchange", "—"),
        "currency":     info.get("currency", "USD"),
        "current_price":round(latest, 2),
        "prev_close":   round(prev, 2),
        "day_change":   round(change, 2),
        "day_change_pct": round(change_pct, 2),
        "market_cap":   info.get("marketCap", None),
        "pe_ratio":     info.get("trailingPE", None),
        "52w_high":     info.get("fiftyTwoWeekHigh", None),
        "52w_low":      info.get("fiftyTwoWeekLow", None),
        "volume":       info.get("volume", None),
        "avg_volume":   info.get("averageVolume", None),
        "dividend_yield": info.get("dividendYield", None),
        "data_start":   str(df.index[0].date()),
        "data_end":     str(df.index[-1].date()),
        "total_rows":   len(df),
    }
