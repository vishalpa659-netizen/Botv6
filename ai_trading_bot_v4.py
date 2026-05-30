import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import sqlite3
import requests
import json
import os
import feedparser
from datetime import datetime, timezone, timedelta
from groq import Groq

# ======================================================
# GUI FILTER BRIDGE  (reads toggles from gui_server.py)
# ======================================================

GUI_FILTER_FILE = "gui_filters.json"

_GUI_FILTER_DEFAULTS = {
    "killzone":       False,
    "calendar":       True,
    "spread":         True,
    "volatility":     True,
    "market_quality": True,
    "regime":         True,
    "mtf":            True,
    "telegram":       True,
}

def read_gui_filters() -> dict:
    try:
        with open(GUI_FILTER_FILE) as f:
            data = json.load(f)
        return {**_GUI_FILTER_DEFAULTS, **data}
    except Exception:
        return dict(_GUI_FILTER_DEFAULTS)

# ======================================================
# CONFIG
# ======================================================

TWELVE_DATA_API_KEY = "7fbe851a56f44979b799a0c75dcbc546"

GROQ_API_KEY     = "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"
TELEGRAM_TOKEN   = "8885360577:AAGDPeUn2drVU1RLNDGJZ91azqMzp0e3QUY"
TELEGRAM_CHAT_ID = "745002829"

# ── EXPANDED SYMBOLS (Twelve Data format) ─────────────────────────────────────
SYMBOLS = [
    "EUR/USD",   # Major
    "GBP/USD",   # Major
    "USD/JPY",   # Major
    "XAU/USD",   # Gold
    "USD/CHF",   # Major
    "AUD/USD",   # Major
    "NZD/USD",   # Major
    "USD/CAD",   # Major
    "GBP/JPY",   # Cross
    "EUR/JPY",   # Cross
    "BTC/USD",   # Crypto
]

SYMBOL_DISPLAY = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "XAU/USD": "GOLD",
    "USD/CHF": "USDCHF",
    "AUD/USD": "AUDUSD",
    "NZD/USD": "NZDUSD",
    "USD/CAD": "USDCAD",
    "GBP/JPY": "GBPJPY",
    "EUR/JPY": "EURJPY",
    "BTC/USD": "BTCUSD",
}

# ── Spread limits per instrument type ─────────────────────────────────────────
SPREAD_LIMITS = {
    "XAU/USD":  1.5,    # Gold — wider spread normal
    "BTC/USD":  200.0,  # BTC — very wide spread normal
    "GBP/JPY":  0.10,
    "EUR/JPY":  0.08,
    "USD/JPY":  0.06,
    "DEFAULT":  0.05,
}

# ── JPY / BTC pip bases for PnL display ───────────────────────────────────────
PIP_BASE = {
    "USDJPY": 0.01,
    "GBPJPY": 0.01,
    "EURJPY": 0.01,
    "GOLD":   0.10,
    "BTCUSD": 1.0,
}

TIMEFRAME        = "5min"
SIGNAL_FILE      = "signal_history.json"
PERFORMANCE_FILE = "performance_data.json"
DATABASE         = "ai_trading.db"

# ======================================================
# UPGRADE 1 — MARKET REGIME CONFIG
# ======================================================

REGIME_ADX_TREND    = 25
REGIME_ADX_VOLATILE = 40

REGIME_MULTIPLIERS = {
    "TRENDING": {"sl": 1.5, "tp": 4.0},
    "RANGING":  {"sl": 1.0, "tp": 1.5},
    "VOLATILE": {"sl": 2.0, "tp": 2.5},
}

# ======================================================
# UPGRADE 3 — SESSION KILLZONE CONFIG (UTC hours)
# ======================================================

KILLZONES = {
    "Asian":    (0,  3),
    "London":   (8,  10),
    "NY_Open":  (13, 15),
    "NY_Close": (19, 20),
}

# ======================================================
# UPGRADE 2 — ECONOMIC CALENDAR CONFIG
# ======================================================

CALENDAR_BLOCK_BEFORE_MIN = 30
CALENDAR_BLOCK_AFTER_MIN  = 15

HIGH_IMPACT_KEYWORDS = [
    "nfp", "non-farm", "interest rate", "rate decision",
    "fomc", "cpi", "inflation", "gdp", "boe", "ecb",
    "rba", "rbnz", "boj", "fed", "powell", "lagarde",
    "unemployment", "payroll"
]

# ======================================================
# UPGRADE 6 — SIGNAL TRACKER CONFIG
# ======================================================

SIGNAL_EXPIRY_HOURS = 24
OPEN_SIGNALS_FILE   = "open_signals.json"

# ======================================================
# NEW — INTELLIGENCE CONFIG
# ======================================================

MIN_PROBABILITY_THRESHOLD = 65   # Don't trade below this confidence
MAX_CONSECUTIVE_LOSSES    = 3    # Circuit breaker: pause after N losses in a row
CIRCUIT_BREAKER_PAUSE_MIN = 30   # How long to pause after circuit breaks (minutes)
MIN_RR_RATIO              = 1.5  # Minimum risk/reward to accept a trade

# ======================================================
# GLOBAL AI STATS
# ======================================================

wins              = 0
losses            = 0
consecutive_losses = 0
circuit_breaker_until = None    # datetime or None

# ======================================================
# CONNECT GROQ
# ======================================================

client = Groq(api_key=GROQ_API_KEY)

# ======================================================
# SQLITE DATABASE
# ======================================================

conn   = sqlite3.connect(DATABASE, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    signal        TEXT,
    probability   REAL,
    trend         TEXT,
    regime        TEXT,
    killzone      TEXT,
    entry         REAL,
    sl            REAL,
    tp            REAL,
    status        TEXT,
    result        TEXT,
    minutes_taken REAL,
    time          TEXT,
    closed_time   TEXT
)
""")

# NEW: per-symbol adaptive learning table
cursor.execute("""
CREATE TABLE IF NOT EXISTS pattern_memory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    regime        TEXT,
    killzone      TEXT,
    rsi_zone      TEXT,
    macd_bullish  INTEGER,
    ema_bullish   INTEGER,
    stoch_zone    TEXT,
    bb_position   TEXT,
    direction     TEXT,
    result        TEXT,
    time          TEXT
)
""")
conn.commit()

# ======================================================
# TELEGRAM
# ======================================================

def _telegram_enabled() -> bool:
    try:
        with open(GUI_FILTER_FILE) as f:
            return json.load(f).get("telegram", True)
    except Exception:
        return True

def send_telegram(message):
    if not _telegram_enabled():
        print(f"[Telegram OFF] {message[:60]}...")
        return
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

# ======================================================
# TWELVE DATA — MARKET DATA
# ======================================================

_price_cache = {}   # symbol → {"bid": x, "ask": x, "time": datetime}

def get_market_data(symbol):
    """Fetch OHLCV candles from Twelve Data. Returns DataFrame or None."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     symbol,
        "interval":   TIMEFRAME,
        "outputsize": 300,
        "apikey":     TWELVE_DATA_API_KEY,
    }
    try:
        r    = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") == "error":
            print(f"Twelve Data Error ({symbol}): {data.get('message')}")
            return None
        values = data.get("values", [])
        if not values:
            print(f"No Data For {symbol}")
            return None
        df = pd.DataFrame(values)
        df = df.rename(columns={
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        })
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        df = df.iloc[::-1].reset_index(drop=True)   # oldest → newest
        return df
    except Exception as e:
        print(f"Market Data Error ({symbol}): {e}")
        return None


def get_live_price(symbol):
    """Get latest bid/ask price from Twelve Data price endpoint."""
    cached = _price_cache.get(symbol)
    if cached:
        age = (datetime.now() - cached["time"]).total_seconds()
        if age < 30:
            return cached["bid"], cached["ask"]

    url = "https://api.twelvedata.com/price"
    params = {"symbol": symbol, "apikey": TWELVE_DATA_API_KEY}
    try:
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        price = float(data.get("price", 0))
        # Estimate spread based on asset type
        if "BTC" in symbol:
            spread_est = price * 0.0005   # 0.05% for crypto
        elif "XAU" in symbol:
            spread_est = price * 0.0002
        else:
            spread_est = price * 0.0001
        bid = price - spread_est / 2
        ask = price + spread_est / 2
        _price_cache[symbol] = {"bid": bid, "ask": ask, "time": datetime.now()}
        return bid, ask
    except Exception as e:
        print(f"Price Error ({symbol}): {e}")
        return None, None


def spread_filter(symbol):
    """Check spread is within acceptable range."""
    bid, ask = get_live_price(symbol)
    if bid is None:
        return True   # allow through if price unavailable
    spread = abs(ask - bid)
    limit  = SPREAD_LIMITS.get(symbol, SPREAD_LIMITS["DEFAULT"])
    print(f"Spread ({symbol}): {round(spread, 5)} | Limit: {limit}")
    if spread > limit:
        return False
    return True

# ======================================================
# ADD INDICATORS  (expanded with BB, Stoch, Williams %R)
# ======================================================

def add_indicators(df):
    df["EMA9"]        = ta.ema(df["close"], length=9)
    df["EMA20"]       = ta.ema(df["close"], length=20)
    df["EMA50"]       = ta.ema(df["close"], length=50)
    df["RSI"]         = ta.rsi(df["close"], length=14)
    macd              = ta.macd(df["close"])
    df["MACD"]        = macd["MACD_12_26_9"]
    df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
    df["ATR"]         = ta.atr(df["high"], df["low"], df["close"], length=14)
    adx_data          = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"]         = adx_data["ADX_14"]

    # NEW: Bollinger Bands
    try:
        bb = ta.bbands(df["close"], length=20, std=2)
        df["BB_UPPER"] = bb["BBU_20_2.0"]
        df["BB_LOWER"] = bb["BBL_20_2.0"]
        df["BB_MID"]   = bb["BBM_20_2.0"]
    except Exception:
        df["BB_UPPER"] = df["close"] * 1.01
        df["BB_LOWER"] = df["close"] * 0.99
        df["BB_MID"]   = df["close"]

    # NEW: Stochastic Oscillator
    try:
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        df["STOCH_K"] = stoch["STOCHk_14_3_3"]
        df["STOCH_D"] = stoch["STOCHd_14_3_3"]
    except Exception:
        df["STOCH_K"] = 50.0
        df["STOCH_D"] = 50.0

    # NEW: Williams %R
    try:
        df["WILLR"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    except Exception:
        df["WILLR"] = -50.0

    return df

# ======================================================
# TREND ANALYSIS
# ======================================================

def get_trend(latest):
    if latest["EMA20"] > latest["EMA50"]:
        return "BULLISH"
    elif latest["EMA20"] < latest["EMA50"]:
        return "BEARISH"
    return "SIDEWAYS"

# ======================================================
# IMPROVED PROBABILITY ENGINE  (directional scoring)
# ======================================================

def calculate_probability(latest, direction_hint, sym_display):
    """
    Direction-aware probability scoring using 7 indicators.
    direction_hint = 'BUY' | 'SELL' | 'HOLD'
    Returns an integer 30–95.
    """
    if direction_hint == "HOLD":
        return 40

    rsi      = float(latest.get("RSI", 50) or 50)
    macd     = float(latest.get("MACD", 0) or 0)
    macd_sig = float(latest.get("MACD_SIGNAL", 0) or 0)
    ema20    = float(latest.get("EMA20", 0) or 0)
    ema50    = float(latest.get("EMA50", 0) or 0)
    ema9     = float(latest.get("EMA9", 0) or 0)
    close    = float(latest.get("close", 0) or 0)
    stoch_k  = float(latest.get("STOCH_K", 50) or 50)
    stoch_d  = float(latest.get("STOCH_D", 50) or 50)
    bb_upper = float(latest.get("BB_UPPER", close * 1.01) or close * 1.01)
    bb_lower = float(latest.get("BB_LOWER", close * 0.99) or close * 0.99)
    bb_mid   = float(latest.get("BB_MID", close) or close)
    willr    = float(latest.get("WILLR", -50) or -50)

    score = 44  # base

    if direction_hint == "BUY":
        # RSI (max ±12)
        if rsi < 30:
            score += 12   # oversold — strong buy zone
        elif rsi < 40:
            score += 8
        elif rsi < 50:
            score += 4
        elif rsi > 70:
            score -= 10   # overbought — risky to buy
        elif rsi > 60:
            score -= 4

        # EMA stack (max ±12)
        if ema9 > ema20 > ema50:
            score += 12   # full bullish stack
        elif ema20 > ema50:
            score += 7
        else:
            score -= 10   # counter-trend penalty

        # MACD (max ±10)
        if macd > macd_sig and macd > 0:
            score += 10   # bullish cross above zero
        elif macd > macd_sig:
            score += 5    # bullish cross below zero
        else:
            score -= 6

        # Stochastic (max ±10)
        if stoch_k < 20 and stoch_k > stoch_d:
            score += 10   # oversold + turning up
        elif stoch_k < 40 and stoch_k > stoch_d:
            score += 5
        elif stoch_k > 80:
            score -= 8    # overbought
        elif stoch_k > stoch_d:
            score += 3

        # Bollinger Bands (max ±8)
        if close <= bb_lower:
            score += 8    # at lower band — bounce zone
        elif close < bb_mid:
            score += 4
        elif close > bb_upper:
            score -= 6    # extended above upper

        # Williams %R (max ±8)
        if willr < -80:
            score += 8    # oversold
        elif willr < -60:
            score += 4
        elif willr > -20:
            score -= 6    # overbought

    elif direction_hint == "SELL":
        # RSI (max ±12)
        if rsi > 70:
            score += 12   # overbought — strong sell zone
        elif rsi > 60:
            score += 8
        elif rsi > 50:
            score += 4
        elif rsi < 30:
            score -= 10   # oversold — risky to sell
        elif rsi < 40:
            score -= 4

        # EMA stack (max ±12)
        if ema9 < ema20 < ema50:
            score += 12   # full bearish stack
        elif ema20 < ema50:
            score += 7
        else:
            score -= 10

        # MACD (max ±10)
        if macd < macd_sig and macd < 0:
            score += 10
        elif macd < macd_sig:
            score += 5
        else:
            score -= 6

        # Stochastic (max ±10)
        if stoch_k > 80 and stoch_k < stoch_d:
            score += 10
        elif stoch_k > 60 and stoch_k < stoch_d:
            score += 5
        elif stoch_k < 20:
            score -= 8
        elif stoch_k < stoch_d:
            score += 3

        # Bollinger Bands (max ±8)
        if close >= bb_upper:
            score += 8
        elif close > bb_mid:
            score += 4
        elif close < bb_lower:
            score -= 6

        # Williams %R (max ±8)
        if willr > -20:
            score += 8
        elif willr > -40:
            score += 4
        elif willr < -80:
            score -= 6

    # Symbol learning adjustment from DB history
    score += get_probability_adjustment(sym_display)

    return min(max(round(score), 30), 95)

# ======================================================
# MARKET QUALITY FILTER
# ======================================================

def market_quality_filter(df):
    latest = df.iloc[-1]
    if 45 <= latest["RSI"] <= 55:
        return False
    return True

# ======================================================
# LIQUIDITY SWEEP DETECTION
# ======================================================

def liquidity_sweep(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest["high"] > previous["high"] and latest["close"] < previous["high"]:
        return "BUY_SIDE_LIQUIDITY"
    if latest["low"] < previous["low"] and latest["close"] > previous["low"]:
        return "SELL_SIDE_LIQUIDITY"
    return "NO_SWEEP"

# ======================================================
# MARKET STRUCTURE DETECTION
# ======================================================

def market_structure(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest["high"] > previous["high"]:
        return "BULLISH_BOS"
    if latest["low"] < previous["low"]:
        return "BEARISH_BOS"
    return "RANGING"

# ======================================================
# ORDER BLOCK DETECTION
# ======================================================

def detect_order_block(df):
    latest      = df.iloc[-2]
    candle_size = abs(latest["close"] - latest["open"])
    total_range = latest["high"] - latest["low"]
    if total_range == 0:
        return "NO_ORDER_BLOCK"
    body_ratio = candle_size / total_range
    if latest["close"] > latest["open"] and body_ratio > 0.6:
        return "BULLISH_ORDER_BLOCK"
    if latest["close"] < latest["open"] and body_ratio > 0.6:
        return "BEARISH_ORDER_BLOCK"
    return "NO_ORDER_BLOCK"

# ======================================================
# FAIR VALUE GAP DETECTION
# ======================================================

def detect_fvg(df):
    candle1 = df.iloc[-3]
    candle3 = df.iloc[-1]
    if candle1["high"] < candle3["low"]:
        return "BULLISH_FVG"
    if candle1["low"] > candle3["high"]:
        return "BEARISH_FVG"
    return "NO_FVG"

# ======================================================
# KEY LEVELS (support & resistance)
# ======================================================

def find_key_levels(df, lookback=30):
    """Find recent support and resistance from last N candles."""
    recent = df.tail(lookback)
    resistance = float(recent["high"].max())
    support    = float(recent["low"].min())
    return support, resistance

# ======================================================
# LIVE FOREX NEWS SENTIMENT
# ======================================================

def news_sentiment():
    try:
        feed_url      = "https://www.forexfactory.com/ffcal_week_this.xml"
        feed          = feedparser.parse(feed_url)
        bullish_words = ["bullish", "rate hike", "strong", "growth"]
        bearish_words = ["bearish", "recession", "crash", "weak"]
        bullish_score = 0
        bearish_score = 0
        for entry in feed.entries[:10]:
            title = entry.title.lower()
            for word in bullish_words:
                if word in title:
                    bullish_score += 1
            for word in bearish_words:
                if word in title:
                    bearish_score += 1
        if bullish_score > bearish_score:
            return "BULLISH_NEWS"
        if bearish_score > bullish_score:
            return "BEARISH_NEWS"
        return "NEUTRAL_NEWS"
    except Exception as e:
        print("News Error:", e)
        return "NEUTRAL_NEWS"

# ======================================================
# SESSION FILTER
# ======================================================

def is_market_session_active():
    current_hour = datetime.now().hour
    if 0 <= current_hour <= 23:
        return True
    return False

# ======================================================
# VOLATILITY FILTER
# ======================================================

def volatility_filter(df):
    if df["ATR"].iloc[-1] <= 0:
        return False
    return True

# ======================================================
# WIN RATE LEARNING
# ======================================================

def update_winrate(result):
    global wins, losses, consecutive_losses, circuit_breaker_until
    if result == "WIN":
        wins += 1
        consecutive_losses = 0    # reset on win
    elif result == "LOSS":
        losses += 1
        consecutive_losses += 1
        # ── Circuit breaker ──────────────────────────────────────
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            circuit_breaker_until = datetime.now() + timedelta(minutes=CIRCUIT_BREAKER_PAUSE_MIN)
            alert = (
                f"🛑 CIRCUIT BREAKER TRIGGERED\n"
                f"{consecutive_losses} consecutive losses detected.\n"
                f"Bot pausing for {CIRCUIT_BREAKER_PAUSE_MIN} minutes to re-calibrate.\n"
                f"Resuming at: {circuit_breaker_until.strftime('%H:%M:%S UTC')}"
            )
            print(alert)
            send_telegram(alert)
    total = wins + losses
    if total == 0:
        return 0
    winrate = (wins / total) * 100
    print(f"REAL WIN RATE: {round(winrate, 2)}%  |  Streak: {consecutive_losses} L")
    return round(winrate, 2)


def is_circuit_breaker_active():
    global circuit_breaker_until
    if circuit_breaker_until is None:
        return False
    if datetime.now() < circuit_breaker_until:
        remaining = (circuit_breaker_until - datetime.now()).seconds // 60
        print(f"🛑 Circuit breaker active — {remaining} min remaining")
        return True
    # Expired — reset
    circuit_breaker_until = None
    consecutive_losses_reset()
    send_telegram("✅ Circuit breaker released — resuming normal trading")
    return False


def consecutive_losses_reset():
    global consecutive_losses
    consecutive_losses = 0

# ======================================================
# ADAPTIVE LEARNING — PER SYMBOL DB STATS
# ======================================================

def get_symbol_winrate(sym_display):
    """Return (wins, losses, winrate%) from closed signals for this symbol."""
    cursor.execute("""
        SELECT
            SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END),
            SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)
        FROM signals
        WHERE symbol=? AND signal != 'HOLD'
          AND result IN ('WIN', 'LOSS')
    """, (sym_display,))
    row = cursor.fetchone()
    if row and row[0] is not None:
        w = int(row[0] or 0)
        l = int(row[1] or 0)
        total = w + l
        if total >= 5:
            return w, l, round((w / total) * 100, 1)
    return 0, 0, 50.0


def get_probability_adjustment(sym_display):
    """Boost or penalise confidence based on per-symbol history."""
    _, _, wr = get_symbol_winrate(sym_display)
    if wr >= 72:
        return 8     # proven performer — boost
    elif wr >= 60:
        return 4
    elif wr >= 50:
        return 0
    elif wr >= 38:
        return -5    # underperforming — reduce confidence
    else:
        return -10   # struggling pair — heavily penalise


def get_regime_session_adjustment(sym_display, regime, killzone):
    """
    Look at historical win rate for THIS symbol in THIS regime+session combo.
    Returns integer adjustment -10 … +10.
    """
    kz = killzone or "OUTSIDE"
    cursor.execute("""
        SELECT
            SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END),
            SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)
        FROM signals
        WHERE symbol=? AND regime=? AND killzone=?
          AND result IN ('WIN','LOSS')
    """, (sym_display, regime, kz))
    row = cursor.fetchone()
    if row and row[0] is not None:
        w = int(row[0] or 0)
        l = int(row[1] or 0)
        total = w + l
        if total >= 4:
            wr = w / total
            if wr >= 0.72:
                return 10
            elif wr >= 0.58:
                return 5
            elif wr < 0.35:
                return -10
            elif wr < 0.45:
                return -5
    return 0   # not enough data


def record_pattern_memory(sym_display, regime, killzone, latest, direction, result):
    """Store indicator fingerprint alongside trade result for future learning."""
    rsi     = float(latest.get("RSI", 50) or 50)
    macd    = float(latest.get("MACD", 0) or 0)
    macd_s  = float(latest.get("MACD_SIGNAL", 0) or 0)
    ema20   = float(latest.get("EMA20", 0) or 0)
    ema50   = float(latest.get("EMA50", 0) or 0)
    stoch_k = float(latest.get("STOCH_K", 50) or 50)
    close   = float(latest.get("close", 0) or 0)
    bb_mid  = float(latest.get("BB_MID", close) or close)

    rsi_zone   = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
    stoch_zone = "OVERSOLD" if stoch_k < 20 else "OVERBOUGHT" if stoch_k > 80 else "NEUTRAL"
    bb_pos     = "ABOVE_MID" if close > bb_mid else "BELOW_MID"

    try:
        cursor.execute("""
            INSERT INTO pattern_memory
            (symbol, regime, killzone, rsi_zone, macd_bullish, ema_bullish,
             stoch_zone, bb_position, direction, result, time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sym_display, regime, killzone or "OUTSIDE",
            rsi_zone, int(macd > macd_s), int(ema20 > ema50),
            stoch_zone, bb_pos, direction, result, str(datetime.now())
        ))
        conn.commit()
    except Exception as e:
        print(f"Pattern memory write error: {e}")

# ======================================================
# REINFORCEMENT LEARNING ENGINE
# ======================================================

def reinforcement_learning(winrate):
    if winrate >= 70:
        print("AI MODE: AGGRESSIVE")
        return 1.2
    elif winrate >= 50:
        print("AI MODE: NORMAL")
        return 1.0
    else:
        print("AI MODE: DEFENSIVE")
        return 0.8

# ======================================================
# SESSION PERFORMANCE
# ======================================================

def session_performance(history):
    london  = 0
    newyork = 0
    asian   = 0
    for signal in history:
        hour = datetime.fromisoformat(signal["time"]).hour
        if 7 <= hour <= 12:
            london += 1
        elif 13 <= hour <= 18:
            newyork += 1
        else:
            asian += 1
    print("\nSESSION PERFORMANCE\n")
    print(f"London Signals: {london}")
    print(f"New York Signals: {newyork}")
    print(f"Asian Signals: {asian}")

# ======================================================
# AUTO STRATEGY OPTIMIZER
# ======================================================

def strategy_optimizer(winrate):
    if winrate >= 75:
        print("OPTIMIZER: HIGH PERFORMANCE MODE")
        return {"risk_multiplier": 1.5, "confidence_boost": 10}
    elif winrate >= 50:
        print("OPTIMIZER: BALANCED MODE")
        return {"risk_multiplier": 1.0, "confidence_boost": 5}
    else:
        print("OPTIMIZER: SAFE MODE")
        return {"risk_multiplier": 0.7, "confidence_boost": 0}

# ======================================================
# UPGRADE 1 — MARKET REGIME DETECTION
# ======================================================

def detect_market_regime(df):
    latest = df.iloc[-1]
    adx    = latest.get("ADX", 0)
    if adx > REGIME_ADX_VOLATILE:
        regime = "VOLATILE"
    elif adx > REGIME_ADX_TREND:
        regime = "TRENDING"
    else:
        regime = "RANGING"
    print(f"Market Regime: {regime} (ADX {round(adx, 2)})")
    return regime

def should_skip_regime(regime):
    if regime == "VOLATILE":
        print("Regime Filter: VOLATILE market — skipping for safety")
        return True
    return False

# ======================================================
# UPGRADE 2 — ECONOMIC CALENDAR FILTER
# ======================================================

def get_economic_calendar():
    events = []
    try:
        feed_url = "https://www.forexfactory.com/ffcal_week_this.xml"
        feed     = feedparser.parse(feed_url)
        for entry in feed.entries:
            title   = entry.get("title", "").lower()
            is_high = any(kw in title for kw in HIGH_IMPACT_KEYWORDS)
            if is_high:
                events.append({
                    "title":    entry.get("title", ""),
                    "impact":   "HIGH",
                    "time_str": entry.get("ff_time_formatted", "")
                })
    except Exception as e:
        print("Calendar Fetch Error:", e)
    return events

def is_calendar_blocked():
    events = get_economic_calendar()
    now    = datetime.now(timezone.utc).replace(tzinfo=None)
    for event in events:
        time_str = event.get("time_str", "")
        if not time_str:
            continue
        try:
            event_time    = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        except Exception:
            continue
        delta_minutes = (event_time - now).total_seconds() / 60
        if -CALENDAR_BLOCK_AFTER_MIN <= delta_minutes <= CALENDAR_BLOCK_BEFORE_MIN:
            print(f"Calendar Block: '{event['title']}' in {round(delta_minutes)} min")
            return True
    return False

# ======================================================
# UPGRADE 3 — SESSION KILLZONE AI
# ======================================================

def get_active_killzone():
    now_hour = datetime.now(timezone.utc).hour
    for name, (start, end) in KILLZONES.items():
        if start <= now_hour < end:
            return name
    return None

def is_killzone_active():
    kz = get_active_killzone()
    if kz:
        print(f"Killzone Active: {kz}")
        return True
    next_kz, minutes_until = get_next_killzone()
    print(f"No Killzone Active. Next: {next_kz} in {minutes_until} min")
    return False

def get_next_killzone():
    now      = datetime.now(timezone.utc)
    now_mins = now.hour * 60 + now.minute
    best_name    = ""
    best_minutes = 9999
    for name, (start, end) in KILLZONES.items():
        start_mins = start * 60
        if start_mins > now_mins:
            diff = start_mins - now_mins
        else:
            diff = (24 * 60 - now_mins) + start_mins
        if diff < best_minutes:
            best_minutes = diff
            best_name    = name
    return best_name, best_minutes

# ======================================================
# UPGRADE 4 — DYNAMIC TP / SL AI  (support/resistance aware)
# ======================================================

def calculate_dynamic_tp_sl(signal_direction, entry, atr, regime, support=None, resistance=None):
    mults   = REGIME_MULTIPLIERS.get(regime, {"sl": 1.5, "tp": 3.0})
    sl_mult = mults["sl"]
    tp_mult = mults["tp"]

    if signal_direction == "BUY":
        raw_sl = entry - (atr * sl_mult)
        raw_tp = entry + (atr * tp_mult)

        # Place SL just below nearest support if it's tighter
        if support is not None and support < entry:
            support_sl = support - (atr * 0.3)
            if support_sl > raw_sl:   # tighter (safer) SL near structure
                sl = support_sl
            else:
                sl = raw_sl
        else:
            sl = raw_sl

        # Cap TP at resistance if resistance sits between entry and raw_tp
        if resistance is not None and entry < resistance < raw_tp:
            tp = resistance - (atr * 0.2)
        else:
            tp = raw_tp

    elif signal_direction == "SELL":
        raw_sl = entry + (atr * sl_mult)
        raw_tp = entry - (atr * tp_mult)

        if resistance is not None and resistance > entry:
            resistance_sl = resistance + (atr * 0.3)
            if resistance_sl < raw_sl:
                sl = resistance_sl
            else:
                sl = raw_sl
        else:
            sl = raw_sl

        if support is not None and entry > support > raw_tp:
            tp = support + (atr * 0.2)
        else:
            tp = raw_tp

    else:
        sl = 0
        tp = 0

    print(f"Dynamic TP/SL | Regime: {regime} | SL×{sl_mult} TP×{tp_mult}")

    # Enforce minimum RR ratio
    if signal_direction in ("BUY", "SELL") and sl != 0 and tp != 0:
        rr = abs(tp - entry) / max(abs(entry - sl), 0.00001)
        if rr < MIN_RR_RATIO:
            # Extend TP to meet minimum RR
            min_tp_distance = abs(entry - sl) * MIN_RR_RATIO
            if signal_direction == "BUY":
                tp = entry + min_tp_distance
            else:
                tp = entry - min_tp_distance
            print(f"TP extended to meet min RR {MIN_RR_RATIO}")

    return round(sl, 5), round(tp, 5)

# ======================================================
# UPGRADE 5 — DASHBOARD REPORTER
# ======================================================

def build_dashboard_report(symbol, signal, regime, killzone, sym_wins, sym_wr):
    sym_display = SYMBOL_DISPLAY.get(symbol, symbol)
    kz_display  = killzone if killzone else "OUTSIDE KILLZONE"
    if signal["signal"] == "HOLD":
        return (
            f"╔══ AI SIGNAL ENGINE v5 ══╗\n"
            f"NO TRADE — {sym_display}\n"
            f"Regime   : {regime}\n"
            f"Killzone : {kz_display}\n"
            f"Trend    : {signal['trend']}\n"
            f"Prob     : {signal['probability']}%\n"
            f"Sym WR   : {sym_wr}%\n"
            f"╚═══════════════════════╝"
        )
    rr = abs(signal["tp"] - signal["entry"]) / max(abs(signal["entry"] - signal["sl"]), 0.00001)
    return (
        f"╔══ AI SIGNAL ENGINE v5 ══╗\n\n"
        f"Symbol   : {sym_display}\n"
        f"Signal   : {signal['signal']}\n"
        f"Entry    : {signal['entry']}\n"
        f"SL       : {signal['sl']}\n"
        f"TP       : {signal['tp']}\n"
        f"RR       : {round(rr, 2)}\n"
        f"Prob     : {signal['probability']}%\n\n"
        f"── UPGRADE LAYERS ──\n"
        f"Regime   : {regime}\n"
        f"Killzone : {kz_display}\n"
        f"Trend    : {signal['trend']}\n"
        f"MTF      : {signal['mtf_confirmation']}\n"
        f"Sweep    : {signal['sweep']}\n"
        f"Structure: {signal['structure']}\n"
        f"OB       : {signal['order_block']}\n"
        f"FVG      : {signal['fvg']}\n"
        f"News     : {signal['news']}\n\n"
        f"── AI LEARNING ──\n"
        f"Sym W/Rate: {sym_wr}% ({sym_wins} wins)\n"
        f"Consec L : {consecutive_losses}\n\n"
        f"📊 Tracking signal until TP/SL hit...\n"
        f"╚═══════════════════════╝"
    )

# ======================================================
# UPGRADE 6 — REAL SIGNAL TRACKER
# ======================================================

def check_open_signals(open_signals):
    still_open = []
    for sig in open_signals:
        symbol = sig["symbol"]
        if sig["signal"] == "HOLD":
            continue
        try:
            td_symbol = next((k for k, v in SYMBOL_DISPLAY.items() if v == symbol), symbol)
            bid, ask  = get_live_price(td_symbol)
            if bid is None:
                still_open.append(sig)
                continue

            current_price = bid
            entry         = sig["entry"]
            sl            = sig["sl"]
            tp            = sig["tp"]
            direction     = sig["signal"]
            open_time     = datetime.fromisoformat(sig["open_time"])
            now           = datetime.now()
            minutes       = round((now - open_time).total_seconds() / 60, 2)
            hours         = minutes / 60
            result        = "RUNNING"

            if direction == "BUY":
                if current_price >= tp:
                    result = "WIN"
                elif current_price <= sl:
                    result = "LOSS"
            elif direction == "SELL":
                if current_price <= tp:
                    result = "WIN"
                elif current_price >= sl:
                    result = "LOSS"

            if result == "RUNNING" and hours >= SIGNAL_EXPIRY_HOURS:
                result = "EXPIRED"

            if result != "RUNNING":
                cursor.execute("""
                    UPDATE signals
                    SET    status       = ?,
                           result       = ?,
                           minutes_taken = ?,
                           closed_time  = ?
                    WHERE  symbol = ?
                      AND  time   = ?
                """, (
                    "CLOSED", result, minutes, str(now), symbol, sig["time"]
                ))
                conn.commit()
                update_winrate(result)

                # Record pattern for adaptive learning
                latest_snapshot = sig.get("latest_snapshot", {})
                if latest_snapshot:
                    record_pattern_memory(
                        symbol,
                        sig.get("regime", "RANGING"),
                        sig.get("killzone", "OUTSIDE"),
                        latest_snapshot,
                        direction,
                        result
                    )

                pip_base = PIP_BASE.get(symbol, 0.0001)
                pnl_pips = round(abs(current_price - entry) / pip_base, 1)

                emoji = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⏰")
                alert = (
                    f"{emoji} SIGNAL CLOSED — {result}\n\n"
                    f"Symbol   : {symbol}\n"
                    f"Direction: {direction}\n"
                    f"Entry    : {entry}\n"
                    f"Close    : {round(current_price, 5)}\n"
                    f"TP       : {tp}\n"
                    f"SL       : {sl}\n"
                    f"Pips     : {pnl_pips}\n"
                    f"Duration : {minutes} min\n"
                    f"Regime   : {sig.get('regime', 'N/A')}\n"
                    f"Killzone : {sig.get('killzone', 'N/A')}"
                )
                print(alert)
                send_telegram(alert)
                send_scoreboard()
            else:
                still_open.append(sig)

        except Exception as e:
            print(f"Tracker Error ({symbol}): {e}")
            still_open.append(sig)

    return still_open


def send_scoreboard():
    cursor.execute("SELECT result, COUNT(*) FROM signals WHERE signal != 'HOLD' GROUP BY result")
    rows    = cursor.fetchall()
    counts  = {row[0]: row[1] for row in rows}
    total_w = counts.get("WIN",     0)
    total_l = counts.get("LOSS",    0)
    total_r = counts.get("RUNNING", 0)
    expired = counts.get("EXPIRED", 0)
    total   = total_w + total_l
    winrate = round((total_w / total) * 100, 2) if total > 0 else 0.0

    cursor.execute("""
        SELECT symbol,
               SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
               COUNT(*) AS total
        FROM   signals WHERE signal != 'HOLD'
        GROUP  BY symbol
    """)
    sym_rows  = cursor.fetchall()
    sym_lines = ""
    for row in sym_rows:
        sym, w, l, t = row
        wr  = round((w / t) * 100, 1) if t > 0 else 0.0
        sym_lines += f"  {sym:<10} W:{w}  L:{l}  WR:{wr}%\n"

    scoreboard = (
        f"📊 SIGNAL SCOREBOARD\n"
        f"{'─'*28}\n"
        f"✅ Wins    : {total_w}\n"
        f"❌ Losses  : {total_l}\n"
        f"🔄 Running : {total_r}\n"
        f"⏰ Expired : {expired}\n"
        f"🎯 Win Rate: {winrate}%\n"
        f"{'─'*28}\n"
        f"Per Symbol:\n"
        f"{sym_lines}"
        f"{'─'*28}"
    )
    print(scoreboard)
    send_telegram(scoreboard)


def get_tracker_summary():
    cursor.execute("SELECT result, COUNT(*) FROM signals WHERE signal != 'HOLD' GROUP BY result")
    rows   = cursor.fetchall()
    counts = {row[0]: row[1] for row in rows}
    w      = counts.get("WIN",     0)
    l      = counts.get("LOSS",    0)
    r      = counts.get("RUNNING", 0)
    total  = w + l
    wr     = round((w / total) * 100, 1) if total > 0 else 0.0
    return f"Tracker → W:{w} L:{l} Running:{r} | WinRate:{wr}%"

# ======================================================
# MULTI TIMEFRAME CONFIRMATION (Twelve Data)
# ======================================================

def multi_timeframe_confirmation(symbol):
    timeframes = ["5min", "15min", "1h"]
    bullish = 0
    bearish = 0
    for tf in timeframes:
        url    = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     symbol,
            "interval":   tf,
            "outputsize": 100,
            "apikey":     TWELVE_DATA_API_KEY,
        }
        try:
            r    = requests.get(url, params=params, timeout=15)
            data = r.json()
            if data.get("status") == "error":
                continue
            values = data.get("values", [])
            df     = pd.DataFrame(values)
            df["close"] = pd.to_numeric(df["close"])
            df          = df.iloc[::-1].reset_index(drop=True)
            df["EMA20"] = ta.ema(df["close"], length=20)
            df["EMA50"] = ta.ema(df["close"], length=50)
            latest      = df.iloc[-1]
            if latest["EMA20"] > latest["EMA50"]:
                bullish += 1
            else:
                bearish += 1
            time.sleep(0.3)   # avoid rate limit
        except Exception:
            continue

    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    return "NEUTRAL"

# ======================================================
# IMPROVED AI SIGNAL GENERATOR
# ======================================================

def generate_signal(symbol, df, regime, killzone=None):
    latest           = df.iloc[-1]
    trend            = get_trend(latest)
    mtf_confirmation = multi_timeframe_confirmation(symbol)
    sweep            = liquidity_sweep(df)
    structure        = market_structure(df)
    order_block      = detect_order_block(df)
    fvg              = detect_fvg(df)
    news             = news_sentiment()
    support, resistance = find_key_levels(df)

    sym_display      = SYMBOL_DISPLAY.get(symbol, symbol)
    sym_wins, sym_losses, sym_wr = get_symbol_winrate(sym_display)

    entry = float(latest["close"])
    atr   = float(latest["ATR"])

    current_winrate  = update_winrate("RUNNING")
    ai_weight        = reinforcement_learning(current_winrate)
    optimizer        = strategy_optimizer(current_winrate)

    # ── Regime+session learning adjustment ────────────────────────────────────
    regime_adj = get_regime_session_adjustment(sym_display, regime, killzone)

    # ── Build Groq prompt (richer context + explicit rules) ───────────────────
    rsi_val     = round(float(latest.get("RSI", 50) or 50), 2)
    macd_val    = round(float(latest.get("MACD", 0) or 0), 6)
    macds_val   = round(float(latest.get("MACD_SIGNAL", 0) or 0), 6)
    ema20_val   = round(float(latest.get("EMA20", 0) or 0), 5)
    ema50_val   = round(float(latest.get("EMA50", 0) or 0), 5)
    adx_val     = round(float(latest.get("ADX", 0) or 0), 2)
    stoch_k_val = round(float(latest.get("STOCH_K", 50) or 50), 2)
    stoch_d_val = round(float(latest.get("STOCH_D", 50) or 50), 2)
    bb_upper    = round(float(latest.get("BB_UPPER", entry * 1.01) or entry * 1.01), 5)
    bb_lower    = round(float(latest.get("BB_LOWER", entry * 0.99) or entry * 0.99), 5)
    willr_val   = round(float(latest.get("WILLR", -50) or -50), 2)

    macd_cross  = "Bullish cross" if macd_val > macds_val else "Bearish cross"
    ema_stack   = "Bullish (EMA20>EMA50)" if ema20_val > ema50_val else "Bearish (EMA20<EMA50)"
    rsi_state   = "(Oversold)" if rsi_val < 30 else "(Overbought)" if rsi_val > 70 else ""

    prompt = f"""You are an elite algorithmic trading AI specialising in Smart Money Concepts (SMC) and ICT methodology.

PAIR: {sym_display}
═══════════════════════════════════════════
Current Price    : {entry}
Market Regime    : {regime}
Active Session   : {killzone or 'Outside Major Session'}
MTF Trend (5/15/60m): {mtf_confirmation}  ← CRITICAL — signals must align with this

INDICATORS:
  RSI 14        : {rsi_val} {rsi_state}
  MACD          : {macd_val} vs Signal {macds_val} → {macd_cross}
  EMA 20/50     : {ema20_val} / {ema50_val} → {ema_stack}
  Stoch K/D     : {stoch_k_val} / {stoch_d_val}
  BB Upper/Lower: {bb_upper} / {bb_lower} | Price: {entry}
  Williams %R   : {willr_val}
  ADX           : {adx_val} (>25 = trending)

SMC ANALYSIS:
  Liquidity Sweep : {sweep}
  Market Structure: {structure}
  Order Block     : {order_block}
  Fair Value Gap  : {fvg}
  News Sentiment  : {news}

AI HISTORY FOR {sym_display}:
  Win Rate: {sym_wr}% ({sym_wins} wins / {sym_losses} losses)
  Regime+Session adj: {regime_adj:+d}

DECISION RULES (follow strictly):
1. BUY  → only if MTF confirms BULLISH AND majority of indicators are bullish
2. SELL → only if MTF confirms BEARISH AND majority of indicators are bearish
3. HOLD → ADX < 20, conflicting signals, MTF is NEUTRAL, OR low confluence
4. Counter-trend trade only if: extreme RSI (<25 or >75) + structure break + FVG present
5. If symbol win rate < 40% → be extra conservative, prefer HOLD

Reply with EXACTLY one word: BUY, SELL, or HOLD"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_direction = response.choices[0].message.content.strip().upper()
        # Extract just the signal word in case model outputs extra text
        if "BUY" in raw_direction:
            signal_direction = "BUY"
        elif "SELL" in raw_direction:
            signal_direction = "SELL"
        else:
            signal_direction = "HOLD"
    except Exception as e:
        print(f"Groq Error: {e}")
        signal_direction = "HOLD"

    # ── MTF alignment enforcement ─────────────────────────────────────────────
    # If Groq says BUY but MTF says BEARISH (or vice versa) → downgrade to HOLD
    if signal_direction == "BUY" and mtf_confirmation == "BEARISH":
        print(f"MTF Conflict: BUY vs BEARISH MTF — downgraded to HOLD")
        signal_direction = "HOLD"
    elif signal_direction == "SELL" and mtf_confirmation == "BULLISH":
        print(f"MTF Conflict: SELL vs BULLISH MTF — downgraded to HOLD")
        signal_direction = "HOLD"

    # ── Probability scoring ───────────────────────────────────────────────────
    probability = calculate_probability(latest, signal_direction, sym_display)

    # Apply reinforcement learning weight + optimizer boost
    probability = min(
        round((probability * ai_weight) + optimizer["confidence_boost"] + regime_adj),
        95
    )

    # ── Minimum probability gate ──────────────────────────────────────────────
    if signal_direction in ("BUY", "SELL") and probability < MIN_PROBABILITY_THRESHOLD:
        print(f"Low probability ({probability}%) — signal downgraded to HOLD")
        signal_direction = "HOLD"

    # ── Calculate TP/SL with support/resistance awareness ─────────────────────
    sl, tp = calculate_dynamic_tp_sl(
        signal_direction, entry, atr, regime, support, resistance
    )

    # Store a snapshot of key indicator values so we can learn from the outcome
    latest_snapshot = {
        "RSI": rsi_val, "MACD": macd_val, "MACD_SIGNAL": macds_val,
        "EMA20": ema20_val, "EMA50": ema50_val, "STOCH_K": stoch_k_val,
        "close": entry, "BB_MID": float(latest.get("BB_MID", entry) or entry),
    }

    return {
        "symbol":           sym_display,
        "signal":           signal_direction,
        "sweep":            sweep,
        "structure":        structure,
        "order_block":      order_block,
        "fvg":              fvg,
        "news":             news,
        "mtf_confirmation": mtf_confirmation,
        "regime":           regime,
        "open_price":       entry,
        "open_time":        str(datetime.now()),
        "status":           "RUNNING",
        "entry":            round(entry, 5),
        "sl":               sl,
        "tp":               tp,
        "probability":      probability,
        "trend":            trend,
        "time":             str(datetime.now()),
        "sym_wr":           sym_wr,
        "sym_wins":         sym_wins,
        "latest_snapshot":  latest_snapshot,
    }

# ======================================================
# SAVE SIGNAL TO DATABASE
# ======================================================

def save_to_database(signal, killzone=""):
    cursor.execute("""
    INSERT INTO signals (
        symbol, signal, probability, trend,
        regime, killzone, entry, sl, tp,
        status, result, minutes_taken, time, closed_time
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal["symbol"],
        signal["signal"],
        signal["probability"],
        signal["trend"],
        signal.get("regime", ""),
        killzone,
        signal.get("entry", 0),
        signal.get("sl", 0),
        signal.get("tp", 0),
        signal.get("status", "RUNNING"),
        "RUNNING",
        signal.get("minutes_taken", 0),
        signal["time"],
        None
    ))
    conn.commit()

# ======================================================
# LOAD / SAVE HISTORY
# ======================================================

def load_history():
    try:
        with open(SIGNAL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(data):
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_open_signals():
    try:
        with open(OPEN_SIGNALS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_open_signals(data):
    with open(OPEN_SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ======================================================
# LEARNING ENGINE  (enhanced — actual pattern analysis)
# ======================================================

def learning_engine(history):
    if len(history) < 10:
        return
    recent          = history[-20:]
    buy_count       = sum(1 for s in recent if s["signal"] == "BUY")
    sell_count      = sum(1 for s in recent if s["signal"] == "SELL")
    hold_count      = sum(1 for s in recent if s["signal"] == "HOLD")
    high_prob_count = sum(1 for s in recent if s["probability"] >= 75)
    confidence      = (high_prob_count / len(recent)) * 100

    print(f"\n── AI Learning Engine ──────────────────")
    print(f"  Signal mix: BUY={buy_count} SELL={sell_count} HOLD={hold_count}")
    print(f"  High-confidence signals: {high_prob_count}/{len(recent)}")
    print(f"  AI Confidence Score: {round(confidence, 2)}%")

    # Per-symbol learning report from DB
    cursor.execute("""
        SELECT symbol, regime,
               SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) wins,
               SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) losses
        FROM signals WHERE signal != 'HOLD' AND result IN ('WIN','LOSS')
        GROUP BY symbol, regime
        HAVING (wins + losses) >= 3
        ORDER BY symbol, wins DESC
    """)
    rows = cursor.fetchall()
    if rows:
        print("  Best regime per symbol:")
        for sym, reg, w, l in rows:
            wr = round((w / (w + l)) * 100, 1) if (w + l) > 0 else 0
            if wr >= 60:
                print(f"    ✅ {sym} in {reg}: {wr}% ({w}W/{l}L)")
            elif wr < 40:
                print(f"    ⚠️ {sym} in {reg}: {wr}% ({w}W/{l}L) — AI avoiding")
    print(f"────────────────────────────────────────")

# ======================================================
# DAILY REPORT
# ======================================================

def daily_report(history):
    total           = len(history)
    buys            = len([x for x in history if x["signal"] == "BUY"])
    sells           = len([x for x in history if x["signal"] == "SELL"])
    avg_probability = np.mean([x["probability"] for x in history]) if history else 0
    report = (
        f"DAILY AI REPORT\n\n"
        f"Total Signals: {total}\n"
        f"BUY Signals: {buys}\n"
        f"SELL Signals: {sells}\n"
        f"Average Probability: {round(avg_probability, 2)}%"
    )
    print(report)
    send_telegram(report)

# ======================================================
# MAIN LOOP  (v5 — expanded pairs, AI learning, no duplicate trades)
# ======================================================

history      = load_history()
open_signals = load_open_signals()

startup_msg = (
    "🚀 AI Smart Money Signal Engine v5 Started\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "[+] 11 Symbols (Forex + Gold + BTC/USD)\n"
    "[+] No-Duplicate-Trade Guard\n"
    "[+] BB + Stoch + Williams %R indicators\n"
    "[+] Adaptive Per-Symbol AI Learning\n"
    "[+] Min Probability Gate (65%+)\n"
    "[+] MTF Alignment Enforcement\n"
    "[+] Circuit Breaker (3 consec losses)\n"
    "[+] Support/Resistance TP/SL\n"
    "[+] Pattern Memory Engine\n"
    "[+] Regime+Session Win Tracking"
)
print(startup_msg)
send_telegram(startup_msg)

send_scoreboard()

while True:

    try:

        gui = read_gui_filters()

        print("\n==============================")
        print("NEW AI MARKET SCAN STARTED")
        print(get_tracker_summary())
        print("==============================\n")

        # ── Circuit breaker check ──────────────────────────────────────────────
        if is_circuit_breaker_active():
            print("Waiting 5 min (circuit breaker)...")
            time.sleep(300)
            continue

        if open_signals:
            print(f"Tracking {len(open_signals)} open signal(s)...")
            open_signals = check_open_signals(open_signals)
            save_open_signals(open_signals)

        active_killzone = get_active_killzone()

        if gui["killzone"] and not is_killzone_active():
            next_kz, mins_until = get_next_killzone()
            print(f"Outside Killzone. Next: {next_kz} in {mins_until} min. Sleeping 5 min.")
            time.sleep(300)
            continue

        if gui["calendar"] and is_calendar_blocked():
            block_msg = "HIGH IMPACT EVENT NEAR — Skipping scan to avoid slippage."
            print(block_msg)
            if gui["telegram"]:
                send_telegram(f"Calendar Block Active: {block_msg}")
            time.sleep(300)
            continue

        # ── Build set of symbols with currently open trades (NO DUPLICATE guard) ──
        open_symbol_set = {
            sig["symbol"] for sig in open_signals
            if sig.get("signal") in ("BUY", "SELL")
        }

        for symbol in SYMBOLS:

            try:

                sym_display = SYMBOL_DISPLAY.get(symbol, symbol)

                # ── SKIP if we already have an open trade on this pair ─────────
                if sym_display in open_symbol_set:
                    print(f"⏸  {sym_display}: Open trade active — skipping new entry")
                    continue

                print(f"\nChecking {sym_display}...")

                df = get_market_data(symbol)

                if df is None:
                    print(f"No data for {sym_display}")
                    continue

                df = add_indicators(df)

                if gui["spread"] and not spread_filter(symbol):
                    print(f"High Spread Skipping {sym_display}")
                    continue

                if gui["volatility"] and not volatility_filter(df):
                    print(f"Low Volatility Skipping {sym_display}")
                    continue

                if gui["market_quality"] and not market_quality_filter(df):
                    print(f"Weak Market Skipping {sym_display}")
                    continue

                regime = detect_market_regime(df)

                if gui["regime"] and should_skip_regime(regime):
                    if gui["telegram"]:
                        send_telegram(f"VOLATILE Regime — {sym_display} skipped")
                    continue

                signal = generate_signal(symbol, df, regime, killzone=active_killzone)

                history.append(signal)
                save_to_database(signal, killzone=active_killzone or "")
                save_history(history)
                learning_engine(history)

                sym_wins = signal.get("sym_wins", 0)
                sym_wr   = signal.get("sym_wr", 50.0)
                dashboard = build_dashboard_report(
                    symbol, signal, regime, active_killzone, sym_wins, sym_wr
                )
                print(dashboard)

                if gui["telegram"]:
                    send_telegram(dashboard)

                if signal["signal"] in ("BUY", "SELL"):
                    signal["killzone"] = active_killzone or ""
                    open_signals.append(signal)
                    save_open_signals(open_signals)
                    # Add to the in-loop set so sibling symbols see the update
                    open_symbol_set.add(sym_display)
                    print(f"Signal added to tracker. Total open: {len(open_signals)}")

                # Small delay between symbols to respect API rate limit
                time.sleep(2)

            except Exception as symbol_error:
                print(f"SYMBOL ERROR ({symbol}):", symbol_error)
                if gui["telegram"]:
                    send_telegram(f"Symbol Error {symbol}: {symbol_error}")
                continue

        print("\nWaiting 5 Minutes...\n")
        time.sleep(300)

    except Exception as e:
        print("MAIN LOOP ERROR:", e)
        send_telegram(f"Main Loop Error: {e}")
        time.sleep(60)
