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
from bytez import Bytez
import yfinance as yf

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

BYTEZ_API_KEY   = "1b3f304bf815e59b9912905961c7a9d9"
TELEGRAM_TOKEN  = "8885360577:AAGDPeUn2drVU1RLNDGJZ91azqMzp0e3QUY"
TELEGRAM_CHAT_ID = "745002829"

# yfinance symbol format
SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "GC=F",      # Gold
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURGBP=X",
    "GBPJPY=X",
    "BTC-USD",   # Bitcoin
]

# Internal display names (for DB / Telegram)
SYMBOL_DISPLAY = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
    "GC=F":     "GOLD",
    "AUDUSD=X": "AUDUSD",
    "USDCAD=X": "USDCAD",
    "USDCHF=X": "USDCHF",
    "NZDUSD=X": "NZDUSD",
    "EURGBP=X": "EURGBP",
    "GBPJPY=X": "GBPJPY",
    "BTC-USD":  "BTCUSD",
}

# Spread limits per symbol (in price units)
SPREAD_LIMITS = {
    "XAU/USD":  1.0,
    "BTC/USD":  50.0,   # BTC has wide spreads
    "GBP/JPY":  0.08,
    "DEFAULT":  0.05,
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
# ONE TRADE PER SYMBOL GUARD
# ======================================================
# Bot will NOT open a new signal on a symbol that already
# has an open/running trade. A new trade is only allowed
# once the previous one is fully CLOSED (WIN/LOSS/EXPIRED).

def symbol_has_open_trade(symbol_display, open_signals):
    """Return True if this symbol already has a live trade."""
    for sig in open_signals:
        if sig.get("symbol") == symbol_display and sig.get("signal") in ("BUY", "SELL"):
            return True
    return False

# ======================================================
# AUTO-LEARNING ENGINE CONFIG
# ======================================================
# Tracks per-symbol win/loss patterns and adjusts minimum
# probability threshold dynamically to avoid repeated losses.

LEARNING_FILE      = "learning_state.json"
MIN_PROB_DEFAULT   = 70    # minimum probability to take a trade
MIN_PROB_FLOOR     = 60    # never go below this
MIN_PROB_CEILING   = 90    # never go above this
LOSS_STREAK_LIMIT  = 3     # consecutive losses before raising threshold

def load_learning_state():
    try:
        with open(LEARNING_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_learning_state(state):
    with open(LEARNING_FILE, "w") as f:
        json.dump(state, f, indent=2)

def update_learning(symbol_display, result):
    """Update per-symbol learning after a trade closes."""
    state = load_learning_state()
    sym   = state.get(symbol_display, {
        "wins": 0, "losses": 0, "streak": 0,
        "min_prob": MIN_PROB_DEFAULT
    })

    if result == "WIN":
        sym["wins"]   += 1
        sym["streak"]  = max(0, sym["streak"] - 1)   # reduce caution on win
        # Slowly lower threshold back toward default on wins
        sym["min_prob"] = max(MIN_PROB_DEFAULT, sym["min_prob"] - 2)

    elif result == "LOSS":
        sym["losses"] += 1
        sym["streak"] += 1
        # Raise bar after each loss, harder after streak
        raise_by = 3 * sym["streak"]
        sym["min_prob"] = min(MIN_PROB_CEILING, sym["min_prob"] + raise_by)
        if sym["streak"] >= LOSS_STREAK_LIMIT:
            print(f"[AutoLearn] {symbol_display} — {sym['streak']} loss streak! "
                  f"Raising min prob to {sym['min_prob']}%")

    total = sym["wins"] + sym["losses"]
    wr    = round(sym["wins"] / total * 100, 1) if total else 0.0
    print(f"[AutoLearn] {symbol_display} → W:{sym['wins']} L:{sym['losses']} "
          f"WR:{wr}% | MinProb:{sym['min_prob']}%")

    state[symbol_display] = sym
    save_learning_state(state)
    return sym

def get_min_probability(symbol_display):
    """Return the current learned minimum probability for this symbol."""
    state = load_learning_state()
    return state.get(symbol_display, {}).get("min_prob", MIN_PROB_DEFAULT)

# ======================================================
# GLOBAL AI STATS
# ======================================================

wins   = 0
losses = 0

# ======================================================
# CONNECT BYTEZ
# ======================================================

client = Bytez(BYTEZ_API_KEY)

# ======================================================
# SQLITE DATABASE
# ======================================================

conn   = sqlite3.connect(DATABASE)
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
# TWELVE DATA — MARKET DATA (replaces MT5)
# ======================================================

# Simple in-memory price cache to avoid wasting API calls
_price_cache = {}   # symbol → {"bid": x, "ask": x, "time": datetime}

def get_market_data(symbol):
    """Fetch OHLCV candles from yfinance. Returns DataFrame or None."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="5m")
        if df is None or df.empty:
            print(f"No Data For {symbol}")
            return None
        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        print(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"Market Data Error ({symbol}): {e}")
        return None


def get_live_price(symbol):
    """Get latest price from yfinance."""
    # Use cache if fresh (< 30 seconds old)
    cached = _price_cache.get(symbol)
    if cached:
        age = (datetime.now() - cached["time"]).total_seconds()
        if age < 30:
            return cached["bid"], cached["ask"]
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")
        if df is None or df.empty:
            return None, None
        price = float(df["Close"].iloc[-1])
        spread_est = price * 0.0001   # estimate 1 pip spread
        bid = price - spread_est / 2
        ask = price + spread_est / 2
        _price_cache[symbol] = {"bid": bid, "ask": ask, "time": datetime.now()}
        return bid, ask
    except Exception as e:
        print(f"Price Error ({symbol}): {e}")
        return None, None


def spread_filter(symbol):
    """Check spread is within acceptable range per symbol."""
    bid, ask = get_live_price(symbol)
    if bid is None:
        return True   # allow through if price unavailable
    spread = abs(ask - bid)
    print(f"Spread ({symbol}): {round(spread, 5)}")
    limit = SPREAD_LIMITS.get(symbol, SPREAD_LIMITS["DEFAULT"])
    if spread > limit:
        return False
    return True

# ======================================================
# ADD INDICATORS
# ======================================================

def add_indicators(df):
    df['EMA20']       = ta.ema(df['close'], length=20)
    df['EMA50']       = ta.ema(df['close'], length=50)
    df['RSI']         = ta.rsi(df['close'], length=14)
    macd              = ta.macd(df['close'])
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
    df['ATR']         = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx_data          = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX']         = adx_data['ADX_14']
    return df

# ======================================================
# TREND ANALYSIS
# ======================================================

def get_trend(latest):
    if latest['EMA20'] > latest['EMA50']:
        return "BULLISH"
    elif latest['EMA20'] < latest['EMA50']:
        return "BEARISH"
    return "SIDEWAYS"

# ======================================================
# PROBABILITY ENGINE
# ======================================================

def calculate_probability(latest):
    score = 50

    # RSI scoring — stronger signals at extremes
    rsi = latest['RSI']
    if rsi > 70 or rsi < 30:
        score += 15   # strong overbought/oversold
    elif rsi > 60 or rsi < 40:
        score += 10   # moderate signal

    # EMA trend alignment
    ema_gap = abs(latest['EMA20'] - latest['EMA50'])
    if latest['EMA20'] > latest['EMA50']:
        score += 15
        if ema_gap > latest['EMA50'] * 0.001:   # meaningful gap
            score += 5
    elif latest['EMA20'] < latest['EMA50']:
        score += 5    # bearish trend still gets partial score

    # MACD momentum
    if latest['MACD'] > latest['MACD_SIGNAL']:
        score += 15
        if abs(latest['MACD'] - latest['MACD_SIGNAL']) > 0.0001:
            score += 5   # strong separation
    elif latest['MACD'] < latest['MACD_SIGNAL']:
        score += 5

    # ADX trend strength
    adx = latest.get('ADX', 0)
    if adx > 30:
        score += 10   # strong trend
    elif adx > 20:
        score += 5    # moderate trend

    return min(round(score), 95)

# ======================================================
# MARKET QUALITY FILTER
# ======================================================

def market_quality_filter(df):
    latest = df.iloc[-1]
    if 45 <= latest['RSI'] <= 55:
        return False
    return True

# ======================================================
# LIQUIDITY SWEEP DETECTION
# ======================================================

def liquidity_sweep(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest['high'] > previous['high'] and latest['close'] < previous['high']:
        return "BUY_SIDE_LIQUIDITY"
    if latest['low'] < previous['low'] and latest['close'] > previous['low']:
        return "SELL_SIDE_LIQUIDITY"
    return "NO_SWEEP"

# ======================================================
# MARKET STRUCTURE DETECTION
# ======================================================

def market_structure(df):
    latest   = df.iloc[-1]
    previous = df.iloc[-2]
    if latest['high'] > previous['high']:
        return "BULLISH_BOS"
    if latest['low'] < previous['low']:
        return "BEARISH_BOS"
    return "RANGING"

# ======================================================
# ORDER BLOCK DETECTION
# ======================================================

def detect_order_block(df):
    latest      = df.iloc[-2]
    candle_size = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    if total_range == 0:
        return "NO_ORDER_BLOCK"
    body_ratio = candle_size / total_range
    if latest['close'] > latest['open'] and body_ratio > 0.6:
        return "BULLISH_ORDER_BLOCK"
    if latest['close'] < latest['open'] and body_ratio > 0.6:
        return "BEARISH_ORDER_BLOCK"
    return "NO_ORDER_BLOCK"

# ======================================================
# FAIR VALUE GAP DETECTION
# ======================================================

def detect_fvg(df):
    candle1 = df.iloc[-3]
    candle3 = df.iloc[-1]
    if candle1['high'] < candle3['low']:
        return "BULLISH_FVG"
    if candle1['low'] > candle3['high']:
        return "BEARISH_FVG"
    return "NO_FVG"

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
    if df['ATR'].iloc[-1] <= 0:
        return False
    return True

# ======================================================
# WIN RATE LEARNING
# ======================================================

def update_winrate(result):
    global wins, losses
    if result == "WIN":
        wins += 1
    elif result == "LOSS":
        losses += 1
    total = wins + losses
    if total == 0:
        return 0
    winrate = (wins / total) * 100
    print(f"REAL WIN RATE: {round(winrate, 2)}%")
    return round(winrate, 2)

# ======================================================
# REINFORCEMENT LEARNING ENGINE
# ======================================================

def reinforcement_learning(winrate):
    global wins, losses
    total = wins + losses
    if total == 0:
        # No history yet — use NORMAL mode, don't penalise
        print("AI MODE: NORMAL (no history)")
        return 1.0
    if winrate >= 70:
        print("AI MODE: AGGRESSIVE")
        return 1.2
    elif winrate >= 50:
        print("AI MODE: NORMAL")
        return 1.0
    else:
        print("AI MODE: DEFENSIVE")
        return 0.9   # softened from 0.8 → 0.9

# ======================================================
# SESSION PERFORMANCE
# ======================================================

def session_performance(history):
    london  = 0
    newyork = 0
    asian   = 0
    for signal in history:
        hour = datetime.fromisoformat(signal['time']).hour
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
    global wins, losses
    total = wins + losses
    if total == 0:
        # No history yet — use balanced mode
        print("OPTIMIZER: BALANCED MODE (no history)")
        return {"risk_multiplier": 1.0, "confidence_boost": 5}
    if winrate >= 75:
        print("OPTIMIZER: HIGH PERFORMANCE MODE")
        return {"risk_multiplier": 1.5, "confidence_boost": 10}
    elif winrate >= 50:
        print("OPTIMIZER: BALANCED MODE")
        return {"risk_multiplier": 1.0, "confidence_boost": 5}
    else:
        print("OPTIMIZER: SAFE MODE")
        return {"risk_multiplier": 0.8, "confidence_boost": 3}

# ======================================================
# UPGRADE 1 — MARKET REGIME DETECTION
# ======================================================

def detect_market_regime(df):
    latest = df.iloc[-1]
    adx    = latest.get('ADX', 0)
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
# UPGRADE 4 — DYNAMIC TP / SL AI
# ======================================================

def calculate_dynamic_tp_sl(signal_direction, entry, atr, regime):
    mults   = REGIME_MULTIPLIERS.get(regime, {"sl": 1.5, "tp": 3.0})
    sl_mult = mults["sl"]
    tp_mult = mults["tp"]
    if signal_direction == "BUY":
        sl = entry - (atr * sl_mult)
        tp = entry + (atr * tp_mult)
    elif signal_direction == "SELL":
        sl = entry + (atr * sl_mult)
        tp = entry - (atr * tp_mult)
    else:
        sl = 0
        tp = 0
    print(f"Dynamic TP/SL | Regime: {regime} | SL×{sl_mult} TP×{tp_mult}")
    return round(sl, 5), round(tp, 5)

# ======================================================
# UPGRADE 5 — DASHBOARD REPORTER
# ======================================================

def build_dashboard_report(symbol, signal, regime, killzone):
    sym_display = SYMBOL_DISPLAY.get(symbol, symbol)
    kz_display  = killzone if killzone else "OUTSIDE KILLZONE"
    if signal['signal'] == "HOLD":
        return (
            f"╔══ AI SIGNAL ENGINE v4 ══╗\n"
            f"NO TRADE — {sym_display}\n"
            f"Regime   : {regime}\n"
            f"Killzone : {kz_display}\n"
            f"Trend    : {signal['trend']}\n"
            f"Prob     : {signal['probability']}%\n"
            f"╚═══════════════════════╝"
        )
    rr = abs(signal['tp'] - signal['entry']) / max(abs(signal['entry'] - signal['sl']), 0.00001)
    return (
        f"╔══ AI SIGNAL ENGINE v4 ══╗\n\n"
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
        f"📊 Tracking signal until TP/SL hit...\n"
        f"╚═══════════════════════╝"
    )

# ======================================================
# UPGRADE 6 — REAL SIGNAL TRACKER
# ======================================================

def check_open_signals(open_signals):
    still_open = []
    for sig in open_signals:
        symbol = sig['symbol']
        if sig['signal'] == "HOLD":
            continue
        try:
            # Use Twelve Data live price
            td_symbol = next((k for k, v in SYMBOL_DISPLAY.items() if v == symbol), symbol)
            bid, ask  = get_live_price(td_symbol)
            if bid is None:
                still_open.append(sig)
                continue

            current_price = bid
            entry         = sig['entry']
            sl            = sig['sl']
            tp            = sig['tp']
            direction     = sig['signal']
            open_time     = datetime.fromisoformat(sig['open_time'])
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
                    "CLOSED", result, minutes, str(now), symbol, sig['time']
                ))
                conn.commit()
                update_winrate(result)
                # ── AUTO-LEARNING: update per-symbol intelligence ──
                if result in ("WIN", "LOSS"):
                    update_learning(symbol, result)

                emoji    = "✅" if result == "WIN" else ("❌" if result == "LOSS" else "⏰")
                if "BTC" in symbol or "BTCUSD" in symbol:
                    pnl_pips = round(abs(current_price - entry), 2)   # BTC in USD
                elif "JPY" in symbol:
                    pnl_pips = round(abs(current_price - entry) / 0.01, 1)  # JPY pairs
                elif "GOLD" in symbol or "XAU" in symbol:
                    pnl_pips = round(abs(current_price - entry), 2)   # Gold in USD
                else:
                    pnl_pips = round(abs(current_price - entry) / 0.0001, 1)  # Forex pips

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
    timeframes = ["5m", "15m", "1h"]
    bullish = 0
    bearish = 0
    for tf in timeframes:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval=tf)
            if df is None or df.empty:
                continue
            df["close"] = pd.to_numeric(df["Close"])
            df = df.reset_index(drop=True)
            df["EMA20"] = ta.ema(df["close"], length=20)
            df["EMA50"] = ta.ema(df["close"], length=50)
            latest = df.iloc[-1]
            if latest["EMA20"] > latest["EMA50"]:
                bullish += 1
            else:
                bearish += 1
            time.sleep(0.3)
        except Exception:
            continue

    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    return "NEUTRAL"
# ======================================================
# AI SIGNAL GENERATOR
# ======================================================

def generate_signal(symbol, df, regime):
    latest           = df.iloc[-1]
    sym_display      = SYMBOL_DISPLAY.get(symbol, symbol)
    trend            = get_trend(latest)
    mtf_confirmation = multi_timeframe_confirmation(symbol)
    sweep            = liquidity_sweep(df)
    structure        = market_structure(df)
    order_block      = detect_order_block(df)
    fvg              = detect_fvg(df)
    news             = news_sentiment()
    probability      = calculate_probability(latest)

    current_winrate  = update_winrate("RUNNING")
    ai_weight        = reinforcement_learning(current_winrate)
    optimizer        = strategy_optimizer(current_winrate)

    probability = min(
        round((probability * ai_weight) + optimizer['confidence_boost']),
        95
    )

    # ── AUTO-LEARNING: enforce learned minimum probability ──
    min_prob = get_min_probability(sym_display)
    if probability < min_prob:
        print(f"[AutoLearn] {sym_display} probability {probability}% < learned min {min_prob}% — HOLD forced")
        return {
            "symbol": sym_display, "signal": "HOLD",
            "sweep": sweep, "structure": structure,
            "order_block": order_block, "fvg": fvg, "news": news,
            "mtf_confirmation": mtf_confirmation, "regime": regime,
            "open_price": latest['close'], "open_time": str(datetime.now()),
            "status": "HOLD", "entry": round(float(latest['close']), 5),
            "sl": 0, "tp": 0, "probability": probability,
            "trend": trend, "time": str(datetime.now()),
        }

    entry = latest['close']
    atr   = latest['ATR']

    # Load per-symbol learning summary for AI context
    learn_state = load_learning_state().get(sym_display, {})
    learn_wins   = learn_state.get("wins", 0)
    learn_losses = learn_state.get("losses", 0)
    learn_streak = learn_state.get("streak", 0)

    prompt = f"""
You are an elite forex and crypto signal AI with adaptive intelligence.

Analyze this market carefully:

Symbol: {sym_display}
Current Price: {entry}
Trend: {trend}
Multi Timeframe Confirmation: {mtf_confirmation}
Liquidity Sweep: {sweep}
Market Structure: {structure}
Order Block: {order_block}
Fair Value Gap: {fvg}
News Sentiment: {news}
Market Regime: {regime}
RSI: {latest['RSI']}
EMA20: {latest['EMA20']}
EMA50: {latest['EMA50']}
MACD: {latest['MACD']}
MACD Signal: {latest['MACD_SIGNAL']}
ADX: {latest.get('ADX', 0)}
ATR: {round(float(atr), 5)}
Winning Probability: {probability}%

Historical Performance on {sym_display}:
  Past Wins: {learn_wins}
  Past Losses: {learn_losses}
  Current Loss Streak: {learn_streak}

Rules:
- If loss streak >= 3, be MORE conservative — prefer HOLD.
- Only signal BUY or SELL if ALL key indicators align strongly.
- When in doubt, reply HOLD.

Reply ONLY with one word:

BUY
SELL
HOLD
"""

    model = client.model("meta-llama/Llama-3.3-70B-Instruct")
    result = model.run(prompt)
    signal_direction = (result.output or "HOLD").strip()
    if signal_direction not in ("BUY", "SELL", "HOLD"):
        signal_direction = "HOLD"
    sl, tp = calculate_dynamic_tp_sl(signal_direction, entry, atr, regime)

    return {
        "symbol":           SYMBOL_DISPLAY.get(symbol, symbol),
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
        "time":             str(datetime.now())
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
        signal['symbol'],
        signal['signal'],
        signal['probability'],
        signal['trend'],
        signal.get('regime', ''),
        killzone,
        signal.get('entry', 0),
        signal.get('sl', 0),
        signal.get('tp', 0),
        signal.get('status', 'RUNNING'),
        "RUNNING",
        signal.get('minutes_taken', 0),
        signal['time'],
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
    except:
        return []

def save_history(data):
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_open_signals():
    try:
        with open(OPEN_SIGNALS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_open_signals(data):
    with open(OPEN_SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ======================================================
# LEARNING ENGINE
# ======================================================

def learning_engine(history):
    if len(history) < 10:
        return
    total            = len(history)
    buy_count        = 0
    sell_count       = 0
    high_probability = 0
    for signal in history[-20:]:
        if signal['signal'] == 'BUY':
            buy_count += 1
        if signal['signal'] == 'SELL':
            sell_count += 1
        if signal['probability'] >= 80:
            high_probability += 1
    confidence = (high_probability / min(total, 20)) * 100
    print(f"AI Confidence Score: {round(confidence, 2)}%")
    print(f"BUY Signals: {buy_count}")
    print(f"SELL Signals: {sell_count}")

# ======================================================
# DAILY REPORT
# ======================================================

def daily_report(history):
    total           = len(history)
    buys            = len([x for x in history if x['signal'] == 'BUY'])
    sells           = len([x for x in history if x['signal'] == 'SELL'])
    avg_probability = np.mean([x['probability'] for x in history])
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
# MAIN LOOP  (v4 — Twelve Data, no MT5)
# ======================================================

history      = load_history()
open_signals = load_open_signals()

print("Advanced AI Smart Money Signal Engine v5 Started (yfinance)")
send_telegram(
    "Advanced AI Smart Money Signal Engine v5 Started\n"
    "[+] yfinance API (free, no key needed)\n"
    "[+] 11 Symbols incl. BTCUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP, GBPJPY\n"
    "[+] One Trade Per Symbol Guard (no repeat entries)\n"
    "[+] Auto-Learning Engine (raises min prob after losses)\n"
    "[+] Market Regime Detection\n"
    "[+] Economic Calendar Filter\n"
    "[+] Session Killzone AI\n"
    "[+] Dynamic TP/SL AI\n"
    "[+] Dashboard Reporter\n"
    "[+] Real Signal Tracker"
)

send_scoreboard()

while True:

    try:

        gui = read_gui_filters()

        print("\n==============================")
        print("NEW AI MARKET SCAN STARTED")
        print(get_tracker_summary())
        print("==============================\n")

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

        for symbol in SYMBOLS:

            try:

                sym_display = SYMBOL_DISPLAY.get(symbol, symbol)
                print(f"\nChecking {sym_display}...")

                # ── ONE TRADE PER SYMBOL GUARD ──
                if symbol_has_open_trade(sym_display, open_signals):
                    print(f"[Guard] {sym_display} already has an open trade — skipping until closed.")
                    continue

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

                signal = generate_signal(symbol, df, regime)

                history.append(signal)
                save_to_database(signal, killzone=active_killzone or "")
                save_history(history)
                learning_engine(history)

                dashboard = build_dashboard_report(symbol, signal, regime, active_killzone)
                print(dashboard)

                if gui["telegram"]:
                    send_telegram(dashboard)

                if signal['signal'] in ("BUY", "SELL"):
                    signal['killzone'] = active_killzone or ""
                    open_signals.append(signal)
                    save_open_signals(open_signals)
                    print(f"Signal added to tracker. Total open: {len(open_signals)}")

                # Small delay between symbols to respect API rate limit
                time.sleep(1)

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
