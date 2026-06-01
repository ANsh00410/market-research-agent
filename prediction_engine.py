"""
prediction_engine.py
────────────────────────────────────────────────────────────────
Bulkowski-based Chart Pattern Detection + Groq AI Price Prediction
Based on: "Encyclopedia of Chart Patterns" 2nd Ed. – Thomas N. Bulkowski

Detects:
  Candlestick patterns  → Doji, Hammer, Shooting Star, Engulfing, Harami,
                          Morning/Evening Star, Marubozu, Spinning Top
  Chart patterns        → Double Top/Bottom, Head & Shoulders, Triangles,
                          Flags, Wedges, Rounding patterns, Rectangles
  Price targets         → Bulkowski Measure Rule (pattern height + breakout)
  Confidence %          → dynamically derived from Bulkowski base × trend context
"""

import os
import json
import math
import numpy as np
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from stock_tools import get_stock_sentiment

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ═════════════════════════════════════════════════════════════════
#  DYNAMIC SCORING & CONFIDENCE HELPERS
# ═════════════════════════════════════════════════════════════════


def _sigmoid_confidence(raw_score: float, max_score: float = 200.0) -> float:
    """
    Convert raw additive score (-∞..+∞) to a 35–95% confidence range using sigmoid.
    Stronger signals → higher confidence. Weak signals → lower confidence.
    Every stock gets a unique, meaningful confidence value.
    """
    x = float(raw_score) / max(float(max_score), 1.0)
    sig = 1.0 / (1.0 + math.exp(-5.0 * x))  # steepness=5
    return 35.0 + sig * 60.0  # output range: 35% to 95%


def _score_strength_factor(raw_score: float, max_score: float = 200.0) -> float:
    """
    Maps absolute score to a 0.5–1.5 multiplier for ATR-based price targets.
    Stronger conviction → larger price target move.
    """
    x = abs(float(raw_score)) / max(float(max_score), 1.0)
    sig = 1.0 / (1.0 + math.exp(-4.0 * x))  # gentler sigmoid
    return 0.5 + sig * 1.0  # range: 0.5x to 1.5x


def _detect_trend(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detect the preceding price trend over `lookback` periods.
    Returns 'uptrend', 'downtrend', or 'sideways'.
    """
    if len(df) < lookback:
        return 'sideways'
    closes = df['close'].values[-lookback:]
    x = np.arange(len(closes), dtype=float)
    slope, _ = np.polyfit(x, closes, 1)
    pct_change = slope * float(lookback) / float(closes[0]) * 100.0
    if pct_change > 3.0:
        return 'uptrend'
    elif pct_change < -3.0:
        return 'downtrend'
    return 'sideways'


def _adjust_pattern_confidence(pattern: dict, trend: str) -> float:
    """
    Dynamically adjust Bulkowski base confidence by preceding trend context.
    Bullish reversal patterns at bottom of downtrend get boosted.
    Bearish reversal patterns at top of uptrend get boosted.
    Patterns in sideways markets get slightly reduced.
    Returns the adjusted confidence as a float.
    """
    base_conf = float(pattern.get('confidence', 50))
    ptype = pattern.get('type', 'neutral')

    if ptype == 'bullish':
        if trend == 'downtrend':    # bullish at bottom → strongest signal
            return min(95.0, base_conf * 1.15)
        elif trend == 'uptrend':    # bullish continuation → slight boost
            return min(95.0, base_conf * 1.05)
        else:                        # sideways → less meaningful
            return base_conf * 0.90
    elif ptype == 'bearish':
        if trend == 'uptrend':       # bearish at top → strongest signal
            return min(95.0, base_conf * 1.15)
        elif trend == 'downtrend':   # bearish continuation → slight boost
            return min(95.0, base_conf * 1.05)
        else:                        # sideways → less meaningful
            return base_conf * 0.90
    return base_conf


def _cluster_support_resistance(df: pd.DataFrame, current_price: float, n_clusters: int = 3):
    """
    Calculate support/resistance using pivot-point clustering.
    Groups nearby price levels (within 1% of each other) and weights by touch frequency.
    Returns (support, resistance) as floats.
    """
    if len(df) < 10:
        return round(current_price * 0.95, 2), round(current_price * 1.05, 2)

    last_60 = df.tail(60)
    lows = sorted(last_60['low'].values.tolist())
    highs = sorted(last_60['high'].values.tolist(), reverse=True)

    def _cluster_levels(prices, threshold_pct=0.01):
        """Group prices within threshold_pct of each other, return cluster centers weighted by frequency."""
        if not prices:
            return []
        clusters = []
        current_cluster = [prices[0]]
        for p in prices[1:]:
            if abs(p - current_cluster[0]) / max(abs(current_cluster[0]), 0.01) < threshold_pct:
                current_cluster.append(p)
            else:
                clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [p]
        if current_cluster:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
        # Sort by frequency (most-touched level first)
        clusters.sort(key=lambda c: c[1], reverse=True)
        return clusters

    # Supports: clustered lows below current price
    support_prices = [l for l in lows if l < current_price * 0.99]
    support_clusters = _cluster_levels(support_prices)
    support = support_clusters[0][0] if support_clusters else current_price * 0.95

    # Resistances: clustered highs above current price
    resistance_prices = [h for h in highs if h > current_price * 1.01]
    resistance_clusters = _cluster_levels(sorted(resistance_prices))
    resistance = resistance_clusters[0][0] if resistance_clusters else current_price * 1.05

    return round(float(support), 2), round(float(resistance), 2)


# ═════════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERN DETECTOR
# ═════════════════════════════════════════════════════════════════


def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Detect candlestick patterns from OHLCV dataframe.
    Returns list of detected patterns with name, type (bullish/bearish/neutral),
    candle_index (0 = latest), and description.
    """
    if len(df) < 3:
        return []

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure we have OHLCV
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return []

    patterns = []
    last = len(df) - 1  # index of latest candle

    # ── Helper lambdas ──────────────────────────────────────────
    def body(i):
        return abs(df["close"].iloc[i] - df["open"].iloc[i])

    def full_r(i):
        return df["high"].iloc[i] - df["low"].iloc[i]

    def upper_s(i):
        return df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])

    def lower_s(i):
        return min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]

    def is_bull(i):
        return df["close"].iloc[i] > df["open"].iloc[i]

    def is_bear(i):
        return df["close"].iloc[i] < df["open"].iloc[i]

    def mid(i):
        return (df["open"].iloc[i] + df["close"].iloc[i]) / 2

    # ── 1. DOJI (open ≈ close, body < 10% of range) ──────────
    for i in [last, last - 1]:
        if i < 0:
            continue
        r = full_r(i)
        if r > 0 and body(i) / r < 0.10:
            patterns.append(
                {
                    "name": "Doji",
                    "type": "neutral",
                    "signal": "NEUTRAL",
                    "candle_idx": last - i,
                    "desc": "Open ≈ Close → indecision/reversal signal. "
                    "Reliability: 55–60%. Watch for confirmation next candle.",
                    "confidence": 55,
                    "bulkowski_note": "Doji shows market indecision. "
                    "Most reliable when appearing after a prolonged trend.",
                }
            )
            break

    # ── 2. HAMMER (bullish reversal at bottom of downtrend) ───
    i = last
    if full_r(i) > 0:
        lo_s = lower_s(i)
        up_s = upper_s(i)
        bd = body(i)
        if lo_s >= 2 * bd and up_s <= 0.3 * bd and bd > 0:
            bias = "bullish" if is_bull(i) else "bullish"  # either color is hammer
            patterns.append(
                {
                    "name": "Hammer",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "candle_idx": 0,
                    "desc": "Long lower shadow with small body → bulls rejected sell-off. "
                    "Bulkowski avg rise after confirmation: 32%.",
                    "confidence": 65,
                    "bulkowski_note": "Hammer at support in a downtrend is highly reliable "
                    "(Bulkowski: ~65% bullish accuracy after confirmation).",
                }
            )

    # ── 3. SHOOTING STAR (bearish reversal at top of uptrend) ─
    if full_r(i) > 0:
        lo_s = lower_s(i)
        up_s = upper_s(i)
        bd = body(i)
        if up_s >= 2 * bd and lo_s <= 0.3 * bd and bd > 0:
            patterns.append(
                {
                    "name": "Shooting Star",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "candle_idx": 0,
                    "desc": "Long upper shadow → bulls pushed price up, bears drove it back down. "
                    "Bearish reversal signal.",
                    "confidence": 63,
                    "bulkowski_note": "Shooting Star near resistance in uptrend → "
                    "avg decline 20–25% (Bulkowski).",
                }
            )

    # ── 4. BULLISH ENGULFING ──────────────────────────────────
    if last >= 1:
        p, c = last - 1, last
        if (
            is_bear(p)
            and is_bull(c)
            and df["open"].iloc[c] <= df["close"].iloc[p]
            and df["close"].iloc[c] >= df["open"].iloc[p]
        ):
            patterns.append(
                {
                    "name": "Bullish Engulfing",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "candle_idx": 0,
                    "desc": "Current bullish candle body engulfs previous bearish body. "
                    "Strong reversal signal. Bulkowski avg rise: 26%.",
                    "confidence": 70,
                    "bulkowski_note": "Engulfing patterns at support levels show 70%+ bullish "
                    "accuracy (Bulkowski statistics).",
                }
            )

    # ── 5. BEARISH ENGULFING ──────────────────────────────────
    if last >= 1:
        p, c = last - 1, last
        if (
            is_bull(p)
            and is_bear(c)
            and df["open"].iloc[c] >= df["close"].iloc[p]
            and df["close"].iloc[c] <= df["open"].iloc[p]
        ):
            patterns.append(
                {
                    "name": "Bearish Engulfing",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "candle_idx": 0,
                    "desc": "Current bearish candle body engulfs previous bullish body. "
                    "Strong reversal signal. Bulkowski avg decline: 22%.",
                    "confidence": 68,
                    "bulkowski_note": "Bearish Engulfing at resistance is a high-reliability "
                    "bearish reversal (Bulkowski).",
                }
            )

    # ── 6. BULLISH HARAMI ─────────────────────────────────────
    if last >= 1:
        p, c = last - 1, last
        if (
            is_bear(p)
            and is_bull(c)
            and df["open"].iloc[c] > df["close"].iloc[p]
            and df["close"].iloc[c] < df["open"].iloc[p]
            and body(c) < body(p) * 0.5
        ):
            patterns.append(
                {
                    "name": "Bullish Harami",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "candle_idx": 0,
                    "desc": "Small bullish candle inside large bearish candle → "
                    "momentum slowing, potential reversal.",
                    "confidence": 58,
                    "bulkowski_note": "Harami signals slowing momentum. "
                    "Less reliable than Engulfing; require confirmation.",
                }
            )

    # ── 7. BEARISH HARAMI ─────────────────────────────────────
    if last >= 1:
        p, c = last - 1, last
        if (
            is_bull(p)
            and is_bear(c)
            and df["open"].iloc[c] < df["close"].iloc[p]
            and df["close"].iloc[c] > df["open"].iloc[p]
            and body(c) < body(p) * 0.5
        ):
            patterns.append(
                {
                    "name": "Bearish Harami",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "candle_idx": 0,
                    "desc": "Small bearish candle inside large bullish candle → "
                    "momentum slowing, potential reversal.",
                    "confidence": 56,
                    "bulkowski_note": "Bearish Harami at resistance → "
                    "consider as early warning, not confirmed signal.",
                }
            )

    # ── 8. MORNING STAR (3-candle bullish reversal) ───────────
    if last >= 2:
        a, b, c = last - 2, last - 1, last
        if (
            is_bear(a)
            and body(b) < body(a) * 0.3
            and is_bull(c)
            and df["close"].iloc[c] > mid(a)
        ):
            patterns.append(
                {
                    "name": "Morning Star",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "candle_idx": 0,
                    "desc": "3-candle reversal: large bearish → small body → large bullish. "
                    "Reliable bullish reversal. Bulkowski avg rise: 35%.",
                    "confidence": 72,
                    "bulkowski_note": "Morning Star is one of the most reliable "
                    "3-candle reversal patterns (Bulkowski: ~72% accuracy).",
                }
            )

    # ── 9. EVENING STAR (3-candle bearish reversal) ───────────
    if last >= 2:
        a, b, c = last - 2, last - 1, last
        if (
            is_bull(a)
            and body(b) < body(a) * 0.3
            and is_bear(c)
            and df["close"].iloc[c] < mid(a)
        ):
            patterns.append(
                {
                    "name": "Evening Star",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "candle_idx": 0,
                    "desc": "3-candle reversal: large bullish → small body → large bearish. "
                    "Reliable bearish reversal. Bulkowski avg decline: 28%.",
                    "confidence": 71,
                    "bulkowski_note": "Evening Star at resistance in uptrend → "
                    "highly reliable (Bulkowski: ~71% accuracy).",
                }
            )

    # ── 10. MARUBOZU (strong directional candle) ─────────────
    if full_r(i) > 0:
        if body(i) / full_r(i) > 0.90:
            direction = "bullish" if is_bull(i) else "bearish"
            signal = "BULLISH" if is_bull(i) else "BEARISH"
            patterns.append(
                {
                    "name": f"{'Bullish' if is_bull(i) else 'Bearish'} Marubozu",
                    "type": direction,
                    "signal": signal,
                    "candle_idx": 0,
                    "desc": "Very little shadow → strong directional conviction. "
                    f"{'Bulls' if is_bull(i) else 'Bears'} in complete control.",
                    "confidence": 60,
                    "bulkowski_note": "Marubozu candles indicate strong momentum. "
                    "Often leads to continuation in short term.",
                }
            )

    # ── 11. SPINNING TOP (small body, large shadows) ──────────
    if full_r(i) > 0:
        if body(i) / full_r(i) < 0.25 and upper_s(i) > body(i) and lower_s(i) > body(i):
            if not any(p["name"] == "Doji" for p in patterns):
                patterns.append(
                    {
                        "name": "Spinning Top",
                        "type": "neutral",
                        "signal": "NEUTRAL",
                        "candle_idx": 0,
                        "desc": "Small body with long shadows on both sides → "
                        "indecision between buyers and sellers.",
                        "confidence": 50,
                        "bulkowski_note": "Spinning Top after a trend often signals exhaustion. "
                        "Wait for confirmation.",
                    }
                )

    # ── 12. THREE WHITE SOLDIERS (bullish continuation) ───────
    if last >= 2:
        trio = [last - 2, last - 1, last]
        if all(is_bull(x) for x in trio):
            if all(
                df["close"].iloc[trio[j]] > df["close"].iloc[trio[j - 1]]
                for j in [1, 2]
            ):
                if all(body(x) > full_r(x) * 0.6 for x in trio):
                    patterns.append(
                        {
                            "name": "Three White Soldiers",
                            "type": "bullish",
                            "signal": "BULLISH",
                            "candle_idx": 0,
                            "desc": "Three consecutive bullish candles with higher closes. "
                            "Strong bullish momentum continuation.",
                            "confidence": 74,
                            "bulkowski_note": "Three White Soldiers show sustained buying pressure. "
                            "Avg rise: 37% (Bulkowski).",
                        }
                    )

    # ── 13. THREE BLACK CROWS (bearish continuation) ──────────
    if last >= 2:
        trio = [last - 2, last - 1, last]
        if all(is_bear(x) for x in trio):
            if all(
                df["close"].iloc[trio[j]] < df["close"].iloc[trio[j - 1]]
                for j in [1, 2]
            ):
                if all(body(x) > full_r(x) * 0.6 for x in trio):
                    patterns.append(
                        {
                            "name": "Three Black Crows",
                            "type": "bearish",
                            "signal": "BEARISH",
                            "candle_idx": 0,
                            "desc": "Three consecutive bearish candles with lower closes. "
                            "Strong bearish momentum continuation.",
                            "confidence": 73,
                            "bulkowski_note": "Three Black Crows show sustained selling pressure. "
                            "Avg decline: 32% (Bulkowski).",
                        }
                    )

    return patterns


# ═════════════════════════════════════════════════════════════════
#  CHART PATTERN DETECTOR  (Bulkowski Measure Rule based)
# ═════════════════════════════════════════════════════════════════


def detect_chart_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Detect multi-day chart patterns using rolling window analysis.
    Based on Bulkowski's Encyclopedia of Chart Patterns (2nd Ed.)
    """
    if len(df) < 20:
        return []

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else None
    n = len(close)

    patterns = []

    # ── HELPER: Find local peaks & troughs ──────────────────
    def find_peaks(arr, window=5):
        peaks = []
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i - window : i + window + 1]):
                peaks.append(i)
        return peaks

    def find_troughs(arr, window=5):
        troughs = []
        for i in range(window, len(arr) - window):
            if arr[i] == min(arr[i - window : i + window + 1]):
                troughs.append(i)
        return troughs

    peaks = find_peaks(close)
    troughs = find_troughs(close)
    cur = close[-1]

    # ── 1. DOUBLE TOP (bearish reversal) ─────────────────────
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if p2 > p1 + 5:  # at least 5 days apart
            h1, h2 = close[p1], close[p2]
            if abs(h1 - h2) / max(h1, h2) < 0.03:  # peaks within 3%
                # Neckline = lowest close between the two peaks
                neckline = min(close[p1 : p2 + 1])
                height = ((h1 + h2) / 2) - neckline
                target = neckline - height  # Bulkowski measure rule
                if cur < neckline * 1.02:  # near or below neckline
                    conf = 72
                    patterns.append(
                        {
                            "name": "Double Top",
                            "type": "bearish",
                            "signal": "BEARISH",
                            "target": round(target, 2),
                            "neckline": round(neckline, 2),
                            "height": round(height, 2),
                            "confidence": conf,
                            "desc": (
                                f"Two peaks at ₹{h1:.0f} & ₹{h2:.0f} → bearish reversal. "
                                f"Neckline: ₹{neckline:.0f}. "
                                f"Bulkowski target: ₹{target:.0f} "
                                f"(avg decline 22% in bull market)."
                            ),
                            "bulkowski_note": (
                                "Double Top: avg decline 22% (bull market), "
                                "24% (bear market). Failure rate ~6% at breakeven. "
                                "Best performance when pattern height > 10% of price."
                            ),
                        }
                    )

    # ── 2. DOUBLE BOTTOM (bullish reversal) ──────────────────
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if t2 > t1 + 5:
            l1, l2 = close[t1], close[t2]
            if abs(l1 - l2) / min(l1, l2) < 0.03:
                neckline = max(close[t1 : t2 + 1])
                height = neckline - ((l1 + l2) / 2)
                target = neckline + height  # Bulkowski measure rule
                if cur > neckline * 0.98:
                    conf = 74
                    patterns.append(
                        {
                            "name": "Double Bottom",
                            "type": "bullish",
                            "signal": "BULLISH",
                            "target": round(target, 2),
                            "neckline": round(neckline, 2),
                            "height": round(height, 2),
                            "confidence": conf,
                            "desc": (
                                f"Two troughs at ₹{l1:.0f} & ₹{l2:.0f} → bullish reversal. "
                                f"Neckline: ₹{neckline:.0f}. "
                                f"Bulkowski target: ₹{target:.0f} "
                                f"(avg rise 37–41% in bull market)."
                            ),
                            "bulkowski_note": (
                                "Double Bottom Eve&Eve: avg rise 37% (bull), 41% (bear). "
                                "Failure rate just 4–6%. Strong pattern especially after "
                                "long downtrend. Volume should be higher on 2nd bottom."
                            ),
                        }
                    )

    # ── 3. HEAD AND SHOULDERS TOP (bearish reversal) ─────────
    if len(peaks) >= 3:
        ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]
        lh, hh, rh = close[ls], close[hd], close[rs]
        if (
            hh > lh * 1.02
            and hh > rh * 1.02  # head higher than shoulders
            and abs(lh - rh) / max(lh, rh) < 0.05
        ):  # shoulders roughly equal
            neckline = (close[ls] + close[rs]) / 2  # simplified neckline
            height = hh - neckline
            target = neckline - height  # Bulkowski measure rule
            conf = 70
            patterns.append(
                {
                    "name": "Head & Shoulders Top",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "target": round(target, 2),
                    "neckline": round(neckline, 2),
                    "height": round(height, 2),
                    "confidence": conf,
                    "desc": (
                        f"Classic H&S: L.Shoulder ₹{lh:.0f} | Head ₹{hh:.0f} | "
                        f"R.Shoulder ₹{rh:.0f}. "
                        f"Neckline: ₹{neckline:.0f}. Target: ₹{target:.0f} "
                        f"(avg decline 22% bull market)."
                    ),
                    "bulkowski_note": (
                        "H&S Top: avg decline 22% (bull), 23% (bear). "
                        "Failure rate 4–7%. Down-sloping necklines show better performance. "
                        "Volume should be highest at left shoulder, declining at head and right."
                    ),
                }
            )

    # ── 4. HEAD AND SHOULDERS BOTTOM / INVERSE H&S (bullish) ─
    if len(troughs) >= 3:
        ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]
        ll, hl, rl = close[ls], close[hd], close[rs]
        if hl < ll * 0.98 and hl < rl * 0.98 and abs(ll - rl) / min(ll, rl) < 0.05:
            neckline = (close[ls] + close[rs]) / 2
            height = neckline - hl
            target = neckline + height
            conf = 72
            patterns.append(
                {
                    "name": "Inverse Head & Shoulders",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "target": round(target, 2),
                    "neckline": round(neckline, 2),
                    "height": round(height, 2),
                    "confidence": conf,
                    "desc": (
                        f"Inverse H&S: L.Sh ₹{ll:.0f} | Head ₹{hl:.0f} | "
                        f"R.Sh ₹{rl:.0f}. "
                        f"Neckline: ₹{neckline:.0f}. Target: ₹{target:.0f} "
                        f"(avg rise 39% bull market)."
                    ),
                    "bulkowski_note": (
                        "Inverse H&S: avg rise 39% (bull), 31% (bear). "
                        "Failure rate just 3–4%. "
                        "Throwback occurs 63% of time — adds buy opportunity."
                    ),
                }
            )

    # ── 5. ASCENDING TRIANGLE (bullish continuation) ─────────
    if n >= 30:
        w = close[-30:]
        highs_30 = high[-30:]
        lows_30 = low[-30:]
        top_resistance = np.percentile(highs_30, 90)
        # Check if lows are rising
        x = np.arange(30)
        slope_l, _ = np.polyfit(x, lows_30, 1)
        if slope_l > 0 and max(highs_30) / min(highs_30) < 1.05:
            height = top_resistance - close[-30]
            target = top_resistance + height  # breakout target
            patterns.append(
                {
                    "name": "Ascending Triangle",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "target": round(target, 2),
                    "neckline": round(top_resistance, 2),
                    "height": round(height, 2),
                    "confidence": 68,
                    "desc": (
                        f"Rising lows meeting flat resistance ₹{top_resistance:.0f} → "
                        f"bullish breakout expected. Target: ₹{target:.0f} "
                        f"(Bulkowski avg rise 35%)."
                    ),
                    "bulkowski_note": (
                        "Ascending Triangle: avg rise 35% (bull market). "
                        "Failure rate 5% at breakeven. "
                        "Breakout day volume improves performance significantly."
                    ),
                }
            )

    # ── 6. DESCENDING TRIANGLE (bearish continuation) ────────
    if n >= 30:
        w = close[-30:]
        highs_30 = high[-30:]
        lows_30 = low[-30:]
        bot_support = np.percentile(lows_30, 10)
        x = np.arange(30)
        slope_h, _ = np.polyfit(x, highs_30, 1)
        if slope_h < 0 and max(lows_30) / min(lows_30) < 1.05:
            height = close[-30] - bot_support
            target = bot_support - height
            patterns.append(
                {
                    "name": "Descending Triangle",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "target": round(target, 2),
                    "neckline": round(bot_support, 2),
                    "height": round(height, 2),
                    "confidence": 66,
                    "desc": (
                        f"Falling highs meeting flat support ₹{bot_support:.0f} → "
                        f"bearish breakdown expected. Target: ₹{target:.0f} "
                        f"(Bulkowski avg decline 19%)."
                    ),
                    "bulkowski_note": (
                        "Descending Triangle: avg decline 19% (bull), 22% (bear). "
                        "Failure rate 3% at breakeven. "
                        "Downward breakouts on heavy volume perform best."
                    ),
                }
            )

    # ── 7. SYMMETRICAL TRIANGLE ────────────────────────────────
    if n >= 20:
        w = 20
        highs_w = high[-w:]
        lows_w = low[-w:]
        x = np.arange(w)
        slope_h, _ = np.polyfit(x, highs_w, 1)
        slope_l, _ = np.polyfit(x, lows_w, 1)
        if slope_h < 0 and slope_l > 0:
            apex = (close[-1] + close[-w]) / 2
            height = highs_w[-w // 2] - lows_w[-w // 2]
            # breakout direction unknown → assess momentum
            mom = close[-1] - close[-5]
            sig = "BULLISH" if mom > 0 else "BEARISH"
            target = (close[-1] + height) if mom > 0 else (close[-1] - height)
            patterns.append(
                {
                    "name": "Symmetrical Triangle",
                    "type": "bullish" if mom > 0 else "bearish",
                    "signal": sig,
                    "target": round(target, 2),
                    "neckline": round(apex, 2),
                    "height": round(height, 2),
                    "confidence": 58,
                    "desc": (
                        f"Converging trend lines → coiling energy. "
                        f"Short-term momentum is {'upward' if mom > 0 else 'downward'}. "
                        f"Breakout target: ₹{target:.0f}."
                    ),
                    "bulkowski_note": (
                        "Symmetrical Triangle: avg rise 31% (bull, up breakout), "
                        "avg decline 18% (downward breakout). "
                        "64% of breakouts are upward in a bull market."
                    ),
                }
            )

    # ── 8. BULL FLAG (sharp rally + consolidation) ────────────
    if n >= 20:
        prev = close[-20:-10]
        recent = close[-10:]
        rally = (prev[-1] - prev[0]) / prev[0]
        consol = (recent[-1] - recent[0]) / recent[0]
        if rally > 0.06 and -0.04 < consol < 0.01:  # strong rally + mild retracement
            flagpole = prev[-1] - prev[0]
            target = close[-1] + flagpole
            patterns.append(
                {
                    "name": "Bull Flag",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "target": round(target, 2),
                    "neckline": round(close[-1], 2),
                    "height": round(flagpole, 2),
                    "confidence": 67,
                    "desc": (
                        f"Sharp +{rally*100:.1f}% rally followed by mild consolidation → "
                        f"continuation likely. Bulkowski target: ₹{target:.0f} "
                        f"(avg rise 23%)."
                    ),
                    "bulkowski_note": (
                        "Flags: avg rise 23% (bull market). Failure rate 4%. "
                        "High-tight flags are the best performing pattern in Bulkowski's data "
                        "with avg rise 69%."
                    ),
                }
            )

    # ── 9. BEAR FLAG (sharp decline + consolidation) ──────────
    if n >= 20:
        prev = close[-20:-10]
        recent = close[-10:]
        drop = (prev[-1] - prev[0]) / prev[0]
        consol = (recent[-1] - recent[0]) / recent[0]
        if drop < -0.06 and -0.01 < consol < 0.04:
            flagpole = prev[0] - prev[-1]
            target = close[-1] - flagpole
            patterns.append(
                {
                    "name": "Bear Flag",
                    "type": "bearish",
                    "signal": "BEARISH",
                    "target": round(target, 2),
                    "neckline": round(close[-1], 2),
                    "height": round(flagpole, 2),
                    "confidence": 65,
                    "desc": (
                        f"Sharp {drop*100:.1f}% decline followed by mild consolidation → "
                        f"continuation lower likely. Target: ₹{target:.0f}."
                    ),
                    "bulkowski_note": (
                        "Bear Flags have avg decline 17% (bull market). "
                        "Heavy volume on the flagpole, light volume on the flag consolidation "
                        "is ideal (Bulkowski)."
                    ),
                }
            )

    # ── 10. ROUNDING BOTTOM (cup/saucer) ──────────────────────
    if n >= 40:
        mid_point = close[n // 2 - 10 : n // 2 + 10]
        start_avg = np.mean(close[:10])
        end_avg = np.mean(close[-10:])
        mid_low = np.min(mid_point)
        if mid_low < start_avg * 0.95 and end_avg > start_avg * 0.98:
            height = end_avg - mid_low
            target = end_avg + height
            patterns.append(
                {
                    "name": "Rounding Bottom",
                    "type": "bullish",
                    "signal": "BULLISH",
                    "target": round(target, 2),
                    "neckline": round(end_avg, 2),
                    "height": round(height, 2),
                    "confidence": 66,
                    "desc": (
                        f"U-shaped price curve → gradual trend reversal from bearish to bullish. "
                        f"Target: ₹{target:.0f} (Bulkowski avg rise 43%)."
                    ),
                    "bulkowski_note": (
                        "Rounding Bottoms: avg rise 43% (best performing bottom pattern). "
                        "Failure rate just 5%. "
                        "Prices often form a 'cup with handle' before the big breakout."
                    ),
                }
            )

    return patterns


# ═════════════════════════════════════════════════════════════════
#  QUANTITATIVE PREDICTION ENGINE  (v2 — Dynamic Confidence)
# ═════════════════════════════════════════════════════════════════

def _get_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Average True Range — measures volatility. Returns float."""
    if len(df) < window + 1:
        return 0.0
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr_val = true_range.rolling(window).mean().iloc[-1]
    return float(atr_val) if not pd.isna(atr_val) else 0.0


# ATR multipliers per timeframe for price target calculation
_ATR_MULTIPLIERS = {
    "swing":  1.5,   # 1-3 days
    "short":  3.0,   # 1-2 weeks
    "medium": 6.0,   # 1-2 months
    "long":   12.0,  # 4-6 months
}


def generate_quantitative_prediction(
    term: str,
    current_price: float,
    stock_data: dict,
    cs_patterns: list,
    chart_patterns: list,
    df: pd.DataFrame,
) -> dict:
    """
    Quantitative prediction engine v2.
    Uses: clustered S/R, trend-context pattern adjustment, sigmoid confidence,
    ATR-scaled targets with score-strength factor, float precision throughout.
    """
    current_price = float(current_price)

    # ── Support / Resistance (clustered pivot points) ─────────
    support, resistance = _cluster_support_resistance(df, current_price)
    # Guard against NaN in S/R
    if pd.isna(support) or support <= 0:
        support = round(current_price * 0.95, 2)
    if pd.isna(resistance) or resistance <= 0:
        resistance = round(current_price * 1.05, 2)

    # ── ATR (volatility measure) ──────────────────────────────
    atr = _get_atr(df)

    # ── RSI (calculated from the actual df, not stock_data) ───
    rsi = 50.0
    if len(df) >= 15:
        d = df["close"].diff()
        g = d.clip(lower=0).rolling(14).mean()
        lo = (-d.clip(upper=0)).rolling(14).mean()
        rs = g / lo.replace(0, float("nan"))
        rsi_val = (100.0 - 100.0 / (1.0 + rs)).iloc[-1]
        rsi = float(rsi_val) if not pd.isna(rsi_val) else 50.0

    # ── Moving Averages (calculated from the actual df) ───────
    ma20_raw = df["close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else float("nan")
    ma20 = float(ma20_raw) if not pd.isna(ma20_raw) else None
    ma50_raw = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else float("nan")
    ma50 = float(ma50_raw) if not pd.isna(ma50_raw) else None

    # ── Trend context for pattern adjustment ──────────────────
    trend = _detect_trend(df)

    # ── Scoring accumulators ──────────────────────────────────
    score = 0.0
    target = None
    primary_pattern = "No pattern"
    detected_patterns = []  # list of names for the orchestrator
    n_bull_signals = 0
    n_bear_signals = 0

    term_lower = term.lower()

    # ── Store chart pattern targets as reference (for capping, NOT primary) ─
    pattern_target_ref = None  # Bulkowski measure-rule target from chart patterns

    # ── Helper: score a pattern with trend-adjusted confidence ─
    def _score_pattern(p, weight=1.0):
        nonlocal score, primary_pattern, n_bull_signals, n_bear_signals, pattern_target_ref
        adj_conf = _adjust_pattern_confidence(p, trend)
        detected_patterns.append(f"{p['name']} ({p['type']}, {adj_conf:.0f}%)")
        if p['type'] == 'bullish':
            score += adj_conf * weight
            n_bull_signals += 1
            primary_pattern = p['name']
            if pattern_target_ref is None and p.get('target'):
                pattern_target_ref = float(p['target'])
        elif p['type'] == 'bearish':
            score -= adj_conf * weight
            n_bear_signals += 1
            primary_pattern = p['name']
            if pattern_target_ref is None and p.get('target'):
                pattern_target_ref = float(p['target'])

    # ═══════════════════════════════════════════════════════════
    #  TIMEFRAME-SPECIFIC SCORING
    # ═══════════════════════════════════════════════════════════

    if "swing" in term_lower:
        timeframe = "1-3 days"
        atr_key = "swing"

        # RSI (mean-reversion is stronger signal for swing)
        if rsi < 30.0:
            score += 25.0; n_bull_signals += 1
        elif rsi < 40.0:
            score += 15.0; n_bull_signals += 1
        elif rsi > 70.0:
            score -= 25.0; n_bear_signals += 1
        elif rsi > 60.0:
            score -= 15.0; n_bear_signals += 1

        # MA proximity (weighted by distance, not just above/below)
        if ma20:
            ma20_dist_pct = (current_price - ma20) / ma20 * 100.0
            score += max(-20.0, min(20.0, ma20_dist_pct * 3.0))
        if ma50:
            ma50_dist_pct = (current_price - ma50) / ma50 * 100.0
            score += max(-12.0, min(12.0, ma50_dist_pct * 1.5))

        # Candlestick patterns (high weight for swing)
        for p in cs_patterns[:3]:
            _score_pattern(p, weight=0.8)
        # Chart patterns
        for p in chart_patterns:
            _score_pattern(p, weight=1.0)

    elif "short" in term_lower:
        timeframe = "1-2 weeks"
        atr_key = "short"

        if rsi < 35.0:
            score += 18.0; n_bull_signals += 1
        elif rsi > 65.0:
            score -= 18.0; n_bear_signals += 1

        if ma20:
            ma20_dist_pct = (current_price - ma20) / ma20 * 100.0
            score += max(-25.0, min(25.0, ma20_dist_pct * 4.0))

        for p in cs_patterns:
            _score_pattern(p, weight=0.5)
        for p in chart_patterns:
            _score_pattern(p, weight=0.8)

    elif "long" in term_lower:
        timeframe = "4-6 months"
        atr_key = "long"

        if ma50:
            ma50_dist_pct = (current_price - ma50) / ma50 * 100.0
            score += max(-35.0, min(35.0, ma50_dist_pct * 3.5))
        if ma20 and ma50:
            if ma20 > ma50:
                score += 10.0; n_bull_signals += 1  # golden cross territory
            else:
                score -= 10.0; n_bear_signals += 1  # death cross territory

        for p in chart_patterns:
            _score_pattern(p, weight=1.0)

    else:  # Medium term
        timeframe = "1-2 months"
        atr_key = "medium"

        if rsi < 40.0:
            score += 12.0; n_bull_signals += 1
        elif rsi > 60.0:
            score -= 12.0; n_bear_signals += 1

        if ma20:
            ma20_dist_pct = (current_price - ma20) / ma20 * 100.0
            score += max(-18.0, min(18.0, ma20_dist_pct * 3.0))
        if ma50:
            ma50_dist_pct = (current_price - ma50) / ma50 * 100.0
            score += max(-18.0, min(18.0, ma50_dist_pct * 2.0))

        for p in cs_patterns[:2]:
            _score_pattern(p, weight=0.4)
        for p in chart_patterns:
            _score_pattern(p, weight=1.0)

    # ═══════════════════════════════════════════════════════════
    #  DIRECTION DETERMINATION (threshold = 10)
    #  Lower threshold gives more directional calls instead of
    #  everything being NEUTRAL. A score of +15 (e.g. RSI=28.6
    #  oversold) should be BULLISH, not NEUTRAL.
    # ═══════════════════════════════════════════════════════════

    if score >= 10.0:
        direction = "BULLISH"
    elif score <= -10.0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # ═══════════════════════════════════════════════════════════
    #  PRICE TARGET — Always ATR-scaled per timeframe
    #  Chart pattern targets are used as directional caps only,
    #  NOT as primary targets. This ensures:
    #    - Swing targets differ from Long targets
    #    - BEARISH targets are always below current price
    #    - BULLISH targets are always above current price
    # ═══════════════════════════════════════════════════════════

    strength = _score_strength_factor(score)
    atr_mult = _ATR_MULTIPLIERS.get(atr_key, 6.0)

    if direction == "BULLISH":
        if atr > 0:
            target = current_price + (atr * atr_mult * strength)
        else:
            target = float(resistance)
        # Cap: don't exceed scaled pattern target (if pattern gave one above price)
        if pattern_target_ref is not None and pattern_target_ref > current_price:
            # Scale pattern target based on timeframe relative to medium term (6.0)
            ratio = atr_mult / 6.0
            scaled_pattern_target = current_price + (pattern_target_ref - current_price) * ratio
            target = min(float(target), float(scaled_pattern_target))
        # Final cap: scale with timeframe (swing=1.5 ATR, long=12 ATR above resistance)
        if atr > 0:
            target = min(float(target), float(resistance) + atr * atr_mult)

    elif direction == "BEARISH":
        if atr > 0:
            target = current_price - (atr * atr_mult * strength)
        else:
            target = float(support)
        # Cap: don't exceed scaled pattern target (if pattern gave one below price)
        if pattern_target_ref is not None and pattern_target_ref < current_price:
            # Scale pattern target based on timeframe relative to medium term (6.0)
            ratio = atr_mult / 6.0
            scaled_pattern_target = current_price + (pattern_target_ref - current_price) * ratio
            target = max(float(target), float(scaled_pattern_target))
        # Final cap: scale with timeframe (swing=1.5 ATR, long=12 ATR below support)
        if atr > 0:
            target = max(float(target), float(support) - atr * atr_mult)

    else:
        # NEUTRAL — give a slight lean based on score sign so the
        # target is not uselessly equal to current_price.
        # Use a fraction of the swing-level ATR displacement.
        if atr > 0 and abs(score) > 1.0:
            lean_mult = 0.5  # half of swing multiplier
            lean_strength = _score_strength_factor(score)
            if score > 0:
                target = current_price + (atr * lean_mult * lean_strength)
            else:
                target = current_price - (atr * lean_mult * lean_strength)
        else:
            target = current_price

    target = round(float(target), 2)
    # Guard against NaN in target
    if pd.isna(target) or target <= 0:
        target = round(current_price, 2)

    # ═══════════════════════════════════════════════════════════
    #  STOP LOSS
    # ═══════════════════════════════════════════════════════════

    if direction == "BULLISH":
        stop_loss = float(support)
    elif direction == "BEARISH":
        stop_loss = float(resistance)
    else:
        # NEUTRAL stop loss: use the closer of support/resistance
        dist_to_sup = abs(current_price - float(support))
        dist_to_res = abs(current_price - float(resistance))
        if score > 0:
            stop_loss = float(support)   # leaning bull, protect downside
        else:
            stop_loss = float(resistance)  # leaning bear, protect upside

    # ═══════════════════════════════════════════════════════════
    #  DYNAMIC CONFIDENCE (sigmoid of absolute score)
    # ═══════════════════════════════════════════════════════════

    # Guard against NaN in stop_loss
    if pd.isna(stop_loss) or stop_loss <= 0:
        stop_loss = current_price

    confidence = _sigmoid_confidence(abs(score))
    # Boost if multiple signals agree
    total_signals = n_bull_signals + n_bear_signals
    if total_signals >= 3:
        confidence = min(95.0, confidence * 1.08)  # +8% for 3+ confirming signals
    elif total_signals >= 2:
        confidence = min(95.0, confidence * 1.04)  # +4% for 2 confirming signals

    confidence = round(confidence, 1)

    # ── Volume confirmation ───────────────────────────────────
    vol_ratio = float(stock_data.get('today_volume', 0)) / max(float(stock_data.get('avg_volume', 1)), 1.0)
    volume_confirmed = vol_ratio > 1.0
    if volume_confirmed and direction != "NEUTRAL":
        confidence = min(95.0, confidence + 2.0)  # small volume bonus

    # ── Risk level ────────────────────────────────────────────
    if "swing" in term_lower:
        risk_level = "HIGH"
    elif confidence >= 75.0:
        risk_level = "LOW"
    elif confidence >= 60.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "direction": direction,
        "confidence": confidence,
        "primary_pattern": primary_pattern,
        "detected_patterns": detected_patterns,
        "price_target": target,
        "price_target_pct": round((target - current_price) / current_price * 100.0, 2),
        "stop_loss": round(stop_loss, 2),
        "timeframe": timeframe,
        "support_level": support,
        "resistance_level": resistance,
        "risk_level": risk_level,
        "volume_confirmation": volume_confirmed,
        # Metadata for the LLM orchestrator (no math from LLM)
        "raw_score": round(score, 2),
        "rsi": round(rsi, 2),
        "ma20": round(ma20, 2) if ma20 else None,
        "ma50": round(ma50, 2) if ma50 else None,
        "atr": round(atr, 2),
        "trend": trend,
        "n_bull_signals": n_bull_signals,
        "n_bear_signals": n_bear_signals,
    }

# ═════════════════════════════════════════════════════════════════
#  MULTI-AGENT ORCHESTRATOR  (v2 — No Math from LLM)
# ═════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """You are an elite AI Trading Orchestrator and Stock Market Expert.
Your job is to synthesize the outputs of two distinct analysis agents into a single strategic verdict:
1. The Quantitative Agent (Technical indicators, chart patterns, price targets, support/resistance)
2. The Sentiment Agent (News sentiment, NLP scoring from recent headlines)

SYNTHESIS RULES:
- If both agents AGREE (e.g., Quant=Bullish, Sentiment=Positive), issue a STRONG directional prediction.
- If Quant has a clear signal but Sentiment is NEUTRAL or MIXED, KEEP the Quant direction. Do not downgrade.
- If the agents strongly DISAGREE (e.g., Quant=Bullish, Sentiment=Negative), issue a CAUTIOUS or NEUTRAL verdict.

IMPORTANT: You decide ONLY the final direction and provide expert reasoning. 
All numerical values (confidence, targets, stop loss) are computed by the Quantitative Engine — do NOT output numbers.
Your reasoning MUST reference specific patterns, indicators, and data points provided below.
Respond ONLY with a valid JSON object.
"""


def get_orchestrated_prediction(
    company: str,
    ticker: str,
    quant_pred: dict,
    sentiment_report: str,
    current_price: float,
) -> dict:
    """
    Multi-Agent Orchestrator v2.
    LLM decides: direction + reasoning ONLY.
    All math (confidence, targets, stop loss) is deterministic Python.
    """
    current_price = float(current_price)

    # Build a rich context prompt with all quant metadata
    detected_pats = quant_pred.get("detected_patterns", [])
    pats_str = ", ".join(detected_pats) if detected_pats else "No chart/candlestick patterns detected"

    rsi_val = quant_pred.get("rsi", 50.0)
    rsi_label = "Oversold" if rsi_val < 35 else ("Overbought" if rsi_val > 65 else "Neutral zone")

    ma20 = quant_pred.get("ma20")
    ma50 = quant_pred.get("ma50")
    ma_info = []
    if ma20:
        pos = "above" if current_price > ma20 else "below"
        dist = abs(current_price - ma20) / ma20 * 100.0
        ma_info.append(f"Price is {dist:.1f}% {pos} MA20 (₹{ma20:,.2f})")
    if ma50:
        pos = "above" if current_price > ma50 else "below"
        dist = abs(current_price - ma50) / ma50 * 100.0
        ma_info.append(f"Price is {dist:.1f}% {pos} MA50 (₹{ma50:,.2f})")
    ma_str = "; ".join(ma_info) if ma_info else "Insufficient data for MA analysis"

    trend = quant_pred.get("trend", "sideways")
    raw_score = quant_pred.get("raw_score", 0)
    n_bull = quant_pred.get("n_bull_signals", 0)
    n_bear = quant_pred.get("n_bear_signals", 0)
    atr_val = quant_pred.get("atr", 0)

    prompt = f"""**Stock:** {company} ({ticker})
**Current Price:** ₹{current_price:,.2f}

**Agent 1: Quantitative Engine Output**
- Direction: {quant_pred['direction']}
- Timeframe: {quant_pred['timeframe']}
- Preceding Trend: {trend}
- Raw Signal Score: {raw_score} (positive=bullish, negative=bearish)
- Bullish Signals: {n_bull} | Bearish Signals: {n_bear}
- RSI(14): {rsi_val:.1f} ({rsi_label})
- Moving Averages: {ma_str}
- ATR(14): ₹{atr_val:,.2f} (daily volatility)
- Detected Patterns: {pats_str}
- Support: ₹{quant_pred.get('support_level', 0):,.2f} | Resistance: ₹{quant_pred.get('resistance_level', 0):,.2f}
- Price Target: ₹{quant_pred.get('price_target', 0):,.2f} ({quant_pred.get('price_target_pct', 0):+.2f}%)

**Agent 2: Sentiment Engine Output**
{sentiment_report}

**Your Task:**
1. Determine the FINAL strategic direction: BULLISH, BEARISH, or NEUTRAL.
2. Write a 5-6 sentence expert analysis explaining your reasoning. Reference specific patterns by name, RSI levels, MA positions, and sentiment findings.

Return EXACTLY this JSON (no extra fields):
{{
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "reasoning": "<Your 5-6 sentence expert analysis>"
}}
"""

    try:
        resp = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.25,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        final_json = json.loads(raw.strip())
        direction = final_json.get("direction", "NEUTRAL").upper()

        # Enforce direction safety
        if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
            direction = "NEUTRAL"

        reasoning = final_json.get("reasoning", "Synthesized using Multi-Agent framework.")

    except Exception as e:
        # Fallback: use quant direction if LLM fails
        direction = quant_pred["direction"]
        reasoning = (
            f"Quant Engine: {quant_pred['direction']} (score {raw_score}) | "
            f"Sentiment: Fallback due to parsing error. Using Quant analysis only."
        )

    # ═══════════════════════════════════════════════════════════
    #  ALL MATH IS DETERMINISTIC PYTHON (no LLM math)
    # ═══════════════════════════════════════════════════════════

    result = {
        "direction": direction,
        "timeframe": quant_pred["timeframe"],
        "reasoning": reasoning,
        "quant_direction": quant_pred["direction"],
        "detected_patterns": quant_pred.get("detected_patterns", []),
    }

    if direction == "NEUTRAL":
        result["price_target"] = round(current_price, 2)
        result["price_target_pct"] = 0.0
        result["stop_loss"] = round(current_price, 2)
        result["support_level"] = quant_pred.get("support_level", current_price)
        result["resistance_level"] = quant_pred.get("resistance_level", current_price)
        result["confidence"] = round(quant_pred.get("confidence", 50.0) * 0.8, 1)  # reduce for neutral
        result["risk_level"] = "MEDIUM"
    elif direction == quant_pred["direction"]:
        # LLM agrees with Quant → inherit all high-precision targets
        result["price_target"] = quant_pred["price_target"]
        result["price_target_pct"] = quant_pred["price_target_pct"]
        result["stop_loss"] = quant_pred["stop_loss"]
        result["support_level"] = quant_pred["support_level"]
        result["resistance_level"] = quant_pred["resistance_level"]
        # Confidence gets a small boost for agreement
        result["confidence"] = min(95.0, round(quant_pred.get("confidence", 50.0) * 1.05, 1))
        result["risk_level"] = quant_pred.get("risk_level", "MEDIUM")
    else:
        # LLM shifted direction → recalculate targets deterministically
        support = quant_pred.get("support_level", current_price * 0.95)
        resistance = quant_pred.get("resistance_level", current_price * 1.05)
        atr = quant_pred.get("atr", 0.0)

        if direction == "BULLISH":
            target = float(resistance)
            stop_loss = float(support)
        else:  # BEARISH
            target = float(support)
            stop_loss = float(resistance)

        result["price_target"] = round(target, 2)
        result["price_target_pct"] = round((target - current_price) / current_price * 100.0, 2)
        result["stop_loss"] = round(stop_loss, 2)
        result["support_level"] = support
        result["resistance_level"] = resistance
        # Reduced confidence since LLM disagrees with quant
        result["confidence"] = round(quant_pred.get("confidence", 50.0) * 0.85, 1)
        result["risk_level"] = "HIGH"

    # Carry over volume confirmation
    result["volume_confirmation"] = quant_pred.get("volume_confirmation", False)

    return result
 
 
# ═════════════════════════════════════════════════════════════════
#  MASTER PREDICTION RUNNER
# ═════════════════════════════════════════════════════════════════
 
 
def run_full_prediction(
    company: str,
    ticker: str,
    stock_data: dict,
    df: pd.DataFrame,
    term: str = "Medium Term",
) -> dict:
    """
    Multi-Agent Full Prediction pipeline:
    1. Quant Agent: Detect candlestick/chart patterns & score indicators
    2. Sentiment Agent: Analyze news sentiment
    3. Orchestrator: Combine outputs
    """
    df_norm = _normalize_df(df)
 
    # 1. Run Quant Agent
    cs_patterns = detect_candlestick_patterns(df_norm)
    chart_patterns = detect_chart_patterns(df_norm)
    current_price = stock_data.get("current_price", 0)
    quant_pred = generate_quantitative_prediction(term, current_price, stock_data, cs_patterns, chart_patterns, df_norm)
 
    # 2. Run Sentiment Agent
    sentiment_report = get_stock_sentiment(company, ticker)
    
    # 3. Run Orchestrator (Passing current_price to calculate targets deterministically)
    orchestrated_pred = get_orchestrated_prediction(company, ticker, quant_pred, sentiment_report, current_price)

    return {
        "prediction": orchestrated_pred,
        "quant_prediction": quant_pred,
        "sentiment_report": sentiment_report,
        "candlestick_patterns": cs_patterns,
        "chart_patterns": chart_patterns,
    }


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names to lowercase and drop rows with NaN values."""
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    if 'close' in df2.columns:
        df2 = df2.dropna(subset=['close'])
    return df2


if __name__ == "__main__":
    import yfinance as yf

    stock = yf.Ticker("RELIANCE.NS")
    df = stock.history(start="2026-01-01")
    if not df.empty:
        snap = {
            "current_price": float(df["Close"].iloc[-1]),
            "today_change": 0.5,
            "rsi": 55.0,
            "ma20": float(df["Close"].rolling(20).mean().iloc[-1]),
            "ma50": None,
            "week52_high": float(df["High"].max()),
            "week52_low": float(df["Low"].min()),
            "return_1m": 3.2,
            "avg_volume": int(df["Volume"].mean()),
            "today_volume": int(df["Volume"].iloc[-1]),
        }
        result = run_full_prediction("Reliance Industries", "RELIANCE.NS", snap, df)
        print(json.dumps(result["prediction"], indent=2))
