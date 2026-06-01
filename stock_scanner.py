"""
stock_scanner.py — AI Stock Recommendation & Scoring Engine
────────────────────────────────────────────────────────────
Scans a list of stocks and scores each one on multiple signals:
  • Bulkowski pattern confidence (from prediction_engine)
  • RSI zone scoring
  • Moving average alignment (trend strength)
  • Momentum (1M, 3M returns)
  • Volume surge (unusual activity)
  • 52W position (proximity to highs/lows)

Outputs ranked BUY and SELL/AVOID recommendations with
entry price, target, stop loss, confidence %, and reasoning.
Also powers the Groq chatbot with full scan context.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ═════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ═════════════════════════════════════════════════════════════════

def get_session():
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session


def score_stock(ticker: str, name: str, mode: str = "eod") -> dict | None:
    """
    Fetch data and compute a composite score (0–100) for a stock.
    
    mode: "intraday" — weights RSI + Volume 2× (quick trades)
          "eod"      — weights MA + Momentum 2× (swing/positional)
    
    Score breakdown:
      RSI zone        : 0–25 pts  (×2 in intraday)
      MA alignment    : 0–20 pts  (×2 in eod)
      Momentum        : 0–20 pts  (×2 in eod)
      Volume surge    : 0–10 pts  (×2 in intraday)
      52W position    : 0–10 pts
      Pattern signal  : 0–15 pts
    """
    # Weight multipliers per mode
    W = {
        "intraday": {"rsi": 2.0, "ma": 0.8, "momentum": 0.8, "volume": 2.0, "pos52": 1.0, "pattern": 1.0},
        "eod":      {"rsi": 1.0, "ma": 2.0,  "momentum": 2.0,  "volume": 0.8, "pos52": 1.0, "pattern": 1.0},
    }.get(mode, {"rsi":1,"ma":1,"momentum":1,"volume":1,"pos52":1,"pattern":1})
    try:
        session = get_session()
        stock = yf.Ticker(ticker, session=session)
        df    = stock.history(period="2y")
        if df is None or df.empty or len(df) < 10:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < 10:
            return None

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        cur    = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if len(close) > 1 else cur
        if cur <= 0 or np.isnan(cur):
            return None

        # ── 1. RSI (14) ──────────────────────────────────────────
        rsi = None
        rsi_score = 10  # neutral
        if len(df) >= 15:
            d = close.diff()
            g = d.clip(lower=0).rolling(14).mean()
            l = (-d.clip(upper=0)).rolling(14).mean()
            r = g / l.replace(0, np.nan)
            v = (100 - 100 / (1 + r)).iloc[-1]
            rsi = round(float(v), 1) if not np.isnan(v) else None

        if rsi:
            if rsi < 25:      rsi_score = 25   # deeply oversold — strong buy signal
            elif rsi < 35:    rsi_score = 22
            elif rsi < 45:    rsi_score = 18
            elif rsi < 55:    rsi_score = 12   # neutral
            elif rsi < 65:    rsi_score = 8
            elif rsi < 75:    rsi_score = 4    # overbought
            else:             rsi_score = 0    # extremely overbought

        # ── 2. MA Alignment ───────────────────────────────────────
        ma20  = float(close.rolling(20).mean().iloc[-1])  if len(df) >= 20 else None
        ma50  = float(close.rolling(50).mean().iloc[-1])  if len(df) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None

        ma_score = 0
        if ma20  and not np.isnan(ma20):
            ma_score += 7 if cur > ma20  else -3
        if ma50  and not np.isnan(ma50):
            ma_score += 8 if cur > ma50  else -4
        if ma200 and not np.isnan(ma200):
            ma_score += 5 if cur > ma200 else -2
        ma_score = max(0, min(20, ma_score + 10))   # normalize 0–20

        # Golden cross / death cross
        golden_cross = death_cross = False
        if ma50 and ma200 and not np.isnan(ma50) and not np.isnan(ma200):
            if len(df) >= 201:
                ma50_prev  = float(close.rolling(50).mean().iloc[-2])
                ma200_prev = float(close.rolling(200).mean().iloc[-2])
                if ma50_prev <= ma200_prev and ma50 > ma200:
                    golden_cross = True
                    ma_score = min(20, ma_score + 5)
                elif ma50_prev >= ma200_prev and ma50 < ma200:
                    death_cross = True
                    ma_score = max(0, ma_score - 5)

        # ── 3. Momentum ───────────────────────────────────────────
        def _ret(days):
            if len(df) < days + 1: return None
            p = float(close.iloc[-days - 1])
            return (cur - p) / p * 100 if p else None

        ret_1m = _ret(21)
        ret_3m = _ret(63)
        ret_1w = _ret(5)

        mom_score = 10  # neutral
        pts = 0
        cnt = 0
        for r in [ret_1w, ret_1m, ret_3m]:
            if r is not None:
                pts += r
                cnt += 1
        if cnt:
            avg_mom = pts / cnt
            if avg_mom > 15:   mom_score = 20
            elif avg_mom > 8:  mom_score = 17
            elif avg_mom > 3:  mom_score = 14
            elif avg_mom > 0:  mom_score = 12
            elif avg_mom > -3: mom_score = 8
            elif avg_mom > -8: mom_score = 5
            else:              mom_score = 2

        # ── 4. Volume surge ───────────────────────────────────────
        vol_score = 5  # neutral
        if len(df) >= 20:
            avg_vol   = float(volume.rolling(20).mean().iloc[-1])
            today_vol = float(volume.iloc[-1])
            if avg_vol > 0 and not np.isnan(avg_vol):
                vol_ratio = today_vol / avg_vol
                if vol_ratio > 3.0:   vol_score = 10
                elif vol_ratio > 2.0: vol_score = 8
                elif vol_ratio > 1.5: vol_score = 7
                elif vol_ratio > 1.0: vol_score = 6
                else:                 vol_score = 4
        else:
            vol_ratio = 1.0

        # ── 5. 52W position ───────────────────────────────────────
        hi52   = float(high.tail(252).max())
        lo52   = float(low.tail(252).min())
        pos52  = (cur - lo52) / (hi52 - lo52) * 100 if (hi52 - lo52) > 0 else 50
        pos52_score = 0
        # Near 52W low = buy opportunity; near 52W high = momentum play
        if pos52 < 10:     pos52_score = 10   # near 52W low — deep value
        elif pos52 < 25:   pos52_score = 9
        elif pos52 < 40:   pos52_score = 7
        elif pos52 < 60:   pos52_score = 6
        elif pos52 < 80:   pos52_score = 7    # upper range — momentum
        elif pos52 < 90:   pos52_score = 8
        else:              pos52_score = 5    # at 52W high — risky

        # ── 6. Candlestick / chart pattern signal ─────────────────
        pattern_score = 7  # neutral
        primary_pattern = "No pattern"
        pattern_type    = "neutral"
        try:
            from prediction_engine import detect_candlestick_patterns, detect_chart_patterns, _normalize_df
            df_norm  = _normalize_df(df)
            cs_pats  = detect_candlestick_patterns(df_norm)
            ch_pats  = detect_chart_patterns(df_norm)
            all_pats = cs_pats + ch_pats

            bull_pts = sum(p.get("confidence", 50) for p in all_pats if p.get("type") == "bullish")
            bear_pts = sum(p.get("confidence", 50) for p in all_pats if p.get("type") == "bearish")

            if all_pats:
                # Pick strongest pattern
                strongest = max(all_pats, key=lambda p: p.get("confidence", 0))
                primary_pattern = strongest["name"]
                pattern_type    = strongest.get("type", "neutral")

            if bull_pts > bear_pts + 30:   pattern_score = 15
            elif bull_pts > bear_pts:      pattern_score = 12
            elif bear_pts > bull_pts + 30: pattern_score = 0
            elif bear_pts > bull_pts:      pattern_score = 3
            else:                          pattern_score = 7
        except Exception:
            pass

        # ── Composite score (weighted by mode) ────────────────────────
        raw_total = (
            rsi_score     * W["rsi"]      +
            ma_score      * W["ma"]       +
            mom_score     * W["momentum"] +
            vol_score     * W["volume"]   +
            pos52_score   * W["pos52"]    +
            pattern_score * W["pattern"]
        )
        max_possible = (25*W["rsi"] + 20*W["ma"] + 20*W["momentum"] +
                        10*W["volume"] + 10*W["pos52"] + 15*W["pattern"])
        total_score  = round((raw_total / max_possible) * 100, 1) if max_possible else 50
        total_score  = max(0.0, min(100.0, total_score))

        # ── Direction & strength ──────────────────────────────────
        if total_score >= 72:    direction, strength = "STRONG BUY",  "🟢🟢🟢"
        elif total_score >= 60:  direction, strength = "BUY",         "🟢🟢"
        elif total_score >= 50:  direction, strength = "WEAK BUY",    "🟢"
        elif total_score >= 42:  direction, strength = "NEUTRAL",     "⬜"
        elif total_score >= 32:  direction, strength = "WEAK SELL",   "🔴"
        elif total_score >= 20:  direction, strength = "SELL",        "🔴🔴"
        else:                    direction, strength = "STRONG SELL", "🔴🔴🔴"

        # ── Price targets (Bulkowski-style) ───────────────────────
        atr = _calc_atr(df, 14)
        if direction in ("STRONG BUY", "BUY", "WEAK BUY"):
            target    = round(cur * (1 + max(0.05, (total_score - 50) / 200)), 2)
            stop_loss = round(cur - 1.5 * atr, 2) if atr else round(cur * 0.95, 2)
        elif direction in ("STRONG SELL", "SELL", "WEAK SELL"):
            target    = round(cur * (1 - max(0.05, (50 - total_score) / 200)), 2)
            stop_loss = round(cur + 1.5 * atr, 2) if atr else round(cur * 1.05, 2)
        else:
            target    = round(cur, 2)
            stop_loss = round(cur * 0.97, 2)

        today_chg = (cur - prev) / prev * 100 if prev else 0

        return {
            "ticker":          ticker,
            "name":            name,
            "score":           round(total_score, 1),
            "direction":       direction,
            "strength":        strength,
            "current_price":   round(cur, 2),
            "today_change":    round(today_chg, 2),
            "target":          target,
            "stop_loss":       stop_loss,
            "rsi":             rsi,
            "ma20":            round(ma20, 2) if ma20 and not np.isnan(ma20) else None,
            "ma50":            round(ma50, 2) if ma50 and not np.isnan(ma50) else None,
            "ma200":           round(ma200, 2) if ma200 and not np.isnan(ma200) else None,
            "ret_1w":          round(ret_1w, 2) if ret_1w is not None else None,
            "ret_1m":          round(ret_1m, 2) if ret_1m is not None else None,
            "ret_3m":          round(ret_3m, 2) if ret_3m is not None else None,
            "vol_ratio":       round(vol_ratio, 2) if vol_ratio else 1.0,
            "hi52":            round(hi52, 2),
            "lo52":            round(lo52, 2),
            "pos52":           round(pos52, 1),
            "primary_pattern": primary_pattern,
            "pattern_type":    pattern_type,
            "golden_cross":    golden_cross,
            "death_cross":     death_cross,
            "score_breakdown": {
                "rsi":     rsi_score,
                "ma":      ma_score,
                "momentum": mom_score,
                "volume":  vol_score,
                "pos52":   pos52_score,
                "pattern": pattern_score,
            }
        }

    except Exception as e:
        return None


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Average True Range for stop loss calculation."""
    try:
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]
        tr    = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════
#  FULL SCANNER
# ═════════════════════════════════════════════════════════════════

def run_scanner(
    stocks: list[tuple[str, str]],
    top_n: int = 10,
    progress_callback=None,
    mode: str = "eod"
) -> dict:
    """
    Scan all stocks, score them, and return ranked recommendations.
    mode: "intraday" or "eod"
    """
    results = []
    total   = len(stocks)

    for idx, (ticker, name) in enumerate(stocks):
        if progress_callback:
            progress_callback(
                (idx + 1) / total,
                f"Scanning {name} ({idx+1}/{total})..."
            )
        result = score_stock(ticker, name, mode=mode)
        if result:
            results.append(result)

    if not results:
        return {"buys": [], "sells": [], "neutral": [], "all": [], "total_scanned": 0}

    # Sort
    results.sort(key=lambda x: x["score"], reverse=True)

    buys    = [r for r in results if r["direction"] in ("STRONG BUY", "BUY", "WEAK BUY")][:top_n]
    sells   = [r for r in results if r["direction"] in ("STRONG SELL", "SELL", "WEAK SELL")]
    sells   = sorted(sells, key=lambda x: x["score"])[:top_n]
    neutral = [r for r in results if r["direction"] == "NEUTRAL"]

    return {
        "buys":          buys,
        "sells":         sells,
        "neutral":       neutral,
        "all":           results,
        "scan_time":     datetime.now().strftime("%H:%M:%S, %d %b %Y"),
        "total_scanned": len(results),
    }


# ═════════════════════════════════════════════════════════════════
#  GROQ RECOMMENDATION CHATBOT
# ═════════════════════════════════════════════════════════════════

CHATBOT_SYSTEM = """You are an expert Indian stock market advisor trained on Bulkowski's pattern analysis
and NSE/BSE market dynamics. You have access to COMPLETE scan data of ALL stocks that were analyzed.

Your role:
- Explain why any stock in the scan is recommended as BUY or SELL in simple, clear language
- If a user asks about a specific stock, find it in the full scan results and explain its score
- Compare stocks when asked
- Explain technical terms (RSI, MA, Bulkowski patterns, 1W/1M/3M returns) simply
- Give risk warnings where appropriate
- Always mention the score and score breakdown when explaining a recommendation
- Use ₹ symbol for prices
- Keep answers concise but informative (3-5 sentences unless asked for more)
- Never give absolute guarantees — always frame as "analysis suggests" or "signals indicate"
- Mention stop loss levels to manage risk
- IMPORTANT: The user may refer to stocks by short names like "Eicher", "Reliance", "TCS" etc.
  Match these to full names in the scan data.

Scoring system:
- Score 72-100: STRONG BUY  — multiple strong bullish signals aligned
- Score 60-71:  BUY         — majority of signals bullish
- Score 50-59:  WEAK BUY    — mild bullish lean
- Score 42-49:  NEUTRAL     — mixed signals
- Score 32-41:  WEAK SELL   — mild bearish lean
- Score 20-31:  SELL        — majority of signals bearish
- Score 0-19:   STRONG SELL — multiple strong bearish signals

Score breakdown (6 signals):
- RSI: oversold (<30) scores high, overbought (>70) scores low
- MA: price above MA20/50/200 scores high
- Momentum: positive 1W/1M/3M returns score high
- Volume: today's volume vs 20-day avg — surge scores high
- 52W position: near 52W low (value buy) or upper range (momentum)
- Pattern: Bulkowski chart/candlestick patterns detected

Always remind users: AI analysis only, not financial advice."""


def chat_with_advisor(
    user_message: str,
    scan_results: dict,
    chat_history: list[dict]
) -> str:
    """
    Chat with the Groq advisor bot with FULL scan context.
    Passes all scanned stocks so the bot can answer about any stock.
    """
    all_stocks = scan_results.get("all", [])
    buys       = scan_results.get("buys", [])
    sells      = scan_results.get("sells", [])

    # ── Format ALL stocks for context (compact 1-line per stock) ──────────
    def _fmt_all(stocks):
        lines = []
        for i, s in enumerate(stocks):
            lines.append(
                f"#{i+1} {s['name']} ({s['ticker'].replace('.NS','')}) | "
                f"Score:{s['score']}/100 | {s['direction']} | "
                f"₹{s['current_price']} | RSI:{s['rsi']} | "
                f"Pattern:{s['primary_pattern']} | "
                f"Target:₹{s['target']} | SL:₹{s['stop_loss']} | "
                f"1M:{s.get('ret_1m','—')}% | "
                f"MA20:{'₹'+str(s['ma20']) if s.get('ma20') else '—'} | "
                f"52W pos:{s.get('pos52','—')}% | "
                f"Vol ratio:{s.get('vol_ratio','—')}x"
            )
        return "\n".join(lines) if lines else "None"

    # Full context with ALL scanned stocks ranked
    context = f"""
=== COMPLETE SCAN RESULTS (as of {scan_results.get('scan_time', 'N/A')}) ===
Total stocks scanned: {scan_results.get('total_scanned', 0)}

ALL STOCKS RANKED BY SCORE (highest = strongest buy signal):
{_fmt_all(all_stocks)}

TOP BUY PICKS: {', '.join(s['name'] for s in buys)}
TOP SELL/AVOID: {', '.join(s['name'] for s in sells)}
"""

    messages = [
        {"role": "system", "content": CHATBOT_SYSTEM + "\n\n" + context}
    ]

    # Add chat history (last 10 messages)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't process that. Error: {str(e)}"


if __name__ == "__main__":
    # Quick test
    test_stocks = [
        ("RELIANCE.NS",  "Reliance Industries"),
        ("HDFCBANK.NS",  "HDFC Bank"),
        ("ZOMATO.NS",    "Zomato"),
        ("BAJFINANCE.NS","Bajaj Finance"),
        ("TCS.NS",       "TCS"),
    ]

    def prog(pct, msg):
        print(f"  [{pct*100:.0f}%] {msg}")

    print("Running scanner...")
    results = run_scanner(test_stocks, top_n=3, progress_callback=prog)
    print(f"\nScanned: {results['total_scanned']} stocks")
    print("\nTOP BUYS:")
    for s in results["buys"]:
        print(f"  {s['name']}: {s['score']}/100 — {s['direction']}")
    print("\nTOP SELLS:")
    for s in results["sells"]:
        print(f"  {s['name']}: {s['score']}/100 — {s['direction']}")
