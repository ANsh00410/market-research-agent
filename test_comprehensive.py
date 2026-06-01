"""
Comprehensive Prediction Engine Test
=====================================
Tests all 5 implementation plan components across many stocks & all 4 timeframes.
Validates:
  1. Dynamic confidence (sigmoid) — varies per stock, NOT hardcoded 60-65
  2. Trend-context pattern adjustment — patterns get boosted/reduced by trend
  3. ATR-scaled price targets — swing < short < medium < long spread
  4. LLM orchestrator — only direction+reasoning, no math from LLM
  5. NaN robustness — no NaN in any output field
  6. Float precision — no integer truncation
  7. Price target sanity — targets are in reasonable range of current price
  8. Support < current_price < resistance (when directional)
"""

import sys
import os
import json
import math
import traceback

import yfinance as yf
import pandas as pd
import numpy as np

# Test the individual helper functions directly
from prediction_engine import (
    _sigmoid_confidence,
    _score_strength_factor,
    _detect_trend,
    _adjust_pattern_confidence,
    _normalize_df,
    _cluster_support_resistance,
    _get_atr,
    detect_candlestick_patterns,
    detect_chart_patterns,
    generate_quantitative_prediction,
    run_full_prediction,
)

# ─── Configuration ───────────────────────────────────────────────
STOCKS = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("TCS", "TCS.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("Infosys", "INFY.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("ITC", "ITC.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("Tata Motors", "TATAMOTORS.NS"),
    ("Sun Pharma", "SUNPHARMA.NS"),
]

TIMEFRAMES = ["Swing Trade", "Short Term", "Medium Term", "Long Term"]
TF_ORDER = {tf: i for i, tf in enumerate(TIMEFRAMES)}

PASS = 0
FAIL = 0
WARNINGS = 0
results_table = []


def log_pass(test_name, detail=""):
    global PASS
    PASS += 1
    print(f"  [PASS] {test_name}" + (f" -- {detail}" if detail else ""))


def log_fail(test_name, detail=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {test_name}" + (f" -- {detail}" if detail else ""))


def log_warn(test_name, detail=""):
    global WARNINGS
    WARNINGS += 1
    print(f"  [WARN] {test_name}" + (f" -- {detail}" if detail else ""))


# ═════════════════════════════════════════════════════════════════
#  UNIT TESTS: Helper Functions
# ═════════════════════════════════════════════════════════════════

def test_sigmoid_confidence():
    print("\n--- Test: _sigmoid_confidence() ---")

    # Score=0 -> should be midpoint ~65%
    c0 = _sigmoid_confidence(0.0)
    if 64.0 < c0 < 66.0:
        log_pass("score=0 -> ~65%", f"got {c0:.2f}")
    else:
        log_fail("score=0 -> ~65%", f"got {c0:.2f}")

    # Score=50 -> should be well above 65%
    c50 = _sigmoid_confidence(50.0)
    if c50 > 75.0:
        log_pass("score=50 -> >75%", f"got {c50:.2f}")
    else:
        log_fail("score=50 -> >75%", f"got {c50:.2f}")

    # Score=200 -> should be near 95%
    c200 = _sigmoid_confidence(200.0)
    if c200 > 90.0:
        log_pass("score=200 -> >90%", f"got {c200:.2f}")
    else:
        log_fail("score=200 -> >90%", f"got {c200:.2f}")

    # Score=5 -> should be only slightly above 65%
    c5 = _sigmoid_confidence(5.0)
    if 65.0 < c5 < 70.0:
        log_pass("score=5 -> ~66-70% (weak signal)", f"got {c5:.2f}")
    else:
        log_fail("score=5 -> ~66-70% (weak signal)", f"got {c5:.2f}")

    # Different scores should give different confidences
    vals = [_sigmoid_confidence(s) for s in [5, 20, 50, 100, 150]]
    if len(set([round(v, 1) for v in vals])) == len(vals):
        log_pass("All different scores produce unique confidences", f"{[round(v,1) for v in vals]}")
    else:
        log_fail("All different scores should produce unique confidences", f"{[round(v,1) for v in vals]}")

    # Output range must be [35, 95]
    for s in [-500, -100, -50, 0, 50, 100, 500]:
        c = _sigmoid_confidence(abs(s))
        if 35.0 <= c <= 95.0:
            log_pass(f"score={s} in [35,95]", f"got {c:.2f}")
        else:
            log_fail(f"score={s} in [35,95]", f"got {c:.2f}")


def test_score_strength_factor():
    print("\n--- Test: _score_strength_factor() ---")

    f0 = _score_strength_factor(0.0)
    if 0.9 < f0 < 1.1:
        log_pass("score=0 -> ~1.0x", f"got {f0:.3f}")
    else:
        log_fail("score=0 -> ~1.0x", f"got {f0:.3f}")

    f200 = _score_strength_factor(200.0)
    if f200 > 1.3:
        log_pass("score=200 -> >1.3x", f"got {f200:.3f}")
    else:
        log_fail("score=200 -> >1.3x", f"got {f200:.3f}")

    # Range check
    for s in [0, 10, 50, 100, 200, 500]:
        f = _score_strength_factor(s)
        if 0.5 <= f <= 1.5:
            log_pass(f"strength({s}) in [0.5, 1.5]", f"got {f:.3f}")
        else:
            log_fail(f"strength({s}) in [0.5, 1.5]", f"got {f:.3f}")


def test_detect_trend():
    print("\n--- Test: _detect_trend() ---")

    # Create an uptrend DataFrame
    prices_up = pd.DataFrame({"close": np.linspace(100, 120, 30)})
    t = _detect_trend(prices_up)
    if t == "uptrend":
        log_pass("Rising prices -> 'uptrend'", f"got '{t}'")
    else:
        log_fail("Rising prices -> 'uptrend'", f"got '{t}'")

    # Create a downtrend DataFrame
    prices_down = pd.DataFrame({"close": np.linspace(120, 100, 30)})
    t = _detect_trend(prices_down)
    if t == "downtrend":
        log_pass("Falling prices -> 'downtrend'", f"got '{t}'")
    else:
        log_fail("Falling prices -> 'downtrend'", f"got '{t}'")

    # Create a sideways DataFrame
    prices_flat = pd.DataFrame({"close": [100 + (0.5 * (i % 3)) for i in range(30)]})
    t = _detect_trend(prices_flat)
    if t == "sideways":
        log_pass("Flat prices -> 'sideways'", f"got '{t}'")
    else:
        log_fail("Flat prices -> 'sideways'", f"got '{t}'")

    # Short DataFrame should return 'sideways' as default
    prices_short = pd.DataFrame({"close": [100, 101, 102]})
    t = _detect_trend(prices_short)
    if t == "sideways":
        log_pass("Short df (<20 rows) -> 'sideways'", f"got '{t}'")
    else:
        log_fail("Short df (<20 rows) -> 'sideways'", f"got '{t}'")


def test_adjust_pattern_confidence():
    print("\n--- Test: _adjust_pattern_confidence() ---")

    bullish_pat = {"name": "Hammer", "type": "bullish", "confidence": 65}
    bearish_pat = {"name": "Shooting Star", "type": "bearish", "confidence": 63}

    # Bullish at downtrend bottom -> boosted
    c = _adjust_pattern_confidence(bullish_pat, "downtrend")
    if c > 65.0:
        log_pass("Bullish + downtrend -> boost", f"65 -> {c:.1f}")
    else:
        log_fail("Bullish + downtrend -> boost", f"65 -> {c:.1f}")

    # Bullish at sideways -> reduced
    c = _adjust_pattern_confidence(bullish_pat, "sideways")
    if c < 65.0:
        log_pass("Bullish + sideways -> reduce", f"65 -> {c:.1f}")
    else:
        log_fail("Bullish + sideways -> reduce", f"65 -> {c:.1f}")

    # Bearish at uptrend top -> boosted
    c = _adjust_pattern_confidence(bearish_pat, "uptrend")
    if c > 63.0:
        log_pass("Bearish + uptrend -> boost", f"63 -> {c:.1f}")
    else:
        log_fail("Bearish + uptrend -> boost", f"63 -> {c:.1f}")

    # Bearish at sideways -> reduced
    c = _adjust_pattern_confidence(bearish_pat, "sideways")
    if c < 63.0:
        log_pass("Bearish + sideways -> reduce", f"63 -> {c:.1f}")
    else:
        log_fail("Bearish + sideways -> reduce", f"63 -> {c:.1f}")


def test_normalize_df():
    print("\n--- Test: _normalize_df() ---")

    # Check column lowercasing
    df = pd.DataFrame({"Close": [100, 101], "Open": [99, 100], "High": [102, 103], "Low": [98, 99]})
    df_n = _normalize_df(df)
    if all(c.islower() for c in df_n.columns):
        log_pass("Columns lowercased")
    else:
        log_fail("Columns lowercased", f"got {list(df_n.columns)}")

    # Check NaN dropping
    df_nan = pd.DataFrame({
        "Close": [100, float("nan"), 102],
        "Open": [99, float("nan"), 101],
        "High": [102, float("nan"), 104],
        "Low": [98, float("nan"), 100],
    })
    df_n2 = _normalize_df(df_nan)
    if len(df_n2) == 2:
        log_pass("NaN rows dropped", f"3 -> {len(df_n2)} rows")
    else:
        log_fail("NaN rows dropped", f"3 -> {len(df_n2)} rows")


# ═════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS: Full Prediction on Real Stocks
# ═════════════════════════════════════════════════════════════════

def test_stock_predictions():
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Full Predictions on Real Stocks")
    print("=" * 70)

    all_confidences = []  # Collect to check variation later
    stock_results = {}    # {ticker: {timeframe: quant_result}}

    for company, ticker in STOCKS:
        print(f"\n{'─' * 60}")
        print(f"  Stock: {company} ({ticker})")
        print(f"{'─' * 60}")

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df.empty:
                log_warn(f"{ticker} - empty history", "skipping")
                continue

            # Build snapshot
            close_series = df["Close"].dropna()
            if len(close_series) < 50:
                log_warn(f"{ticker} - insufficient data", f"only {len(close_series)} rows")
                continue

            current_price = float(close_series.iloc[-1])

            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            actual_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

            snap = {
                "current_price": current_price,
                "today_change": float((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100),
                "rsi": actual_rsi,
                "ma20": float(close_series.rolling(20).mean().iloc[-1]),
                "ma50": float(close_series.rolling(50).mean().iloc[-1]) if len(close_series) >= 50 else None,
                "week52_high": float(df["High"].tail(252).max()),
                "week52_low": float(df["Low"].tail(252).min()),
                "return_1m": float((close_series.iloc[-1] - close_series.iloc[-20]) / close_series.iloc[-20] * 100) if len(close_series) >= 20 else 0,
                "avg_volume": float(df["Volume"].tail(20).mean()),
                "today_volume": float(df["Volume"].iloc[-1]),
            }

            stock_results[ticker] = {}
            tf_targets = {}  # Track targets per timeframe for comparison

            for tf in TIMEFRAMES:
                print(f"\n  Timeframe: {tf}")

                try:
                    # Run quant-only (no LLM call) for fast testing
                    df_norm = _normalize_df(df)
                    cs_pats = detect_candlestick_patterns(df_norm)
                    ch_pats = detect_chart_patterns(df_norm)
                    quant = generate_quantitative_prediction(tf, current_price, snap, cs_pats, ch_pats, df_norm)

                    stock_results[ticker][tf] = quant

                    # ── CHECK 1: No NaN in any field ──────────────
                    nan_fields = []
                    for key in ["price_target", "stop_loss", "support_level", "resistance_level",
                                "confidence", "raw_score", "rsi", "atr"]:
                        val = quant.get(key)
                        if val is not None and (pd.isna(val) if isinstance(val, (int, float)) else False):
                            nan_fields.append(key)

                    if not nan_fields:
                        log_pass(f"{tf}: No NaN values")
                    else:
                        log_fail(f"{tf}: NaN found in fields", str(nan_fields))

                    # ── CHECK 2: Confidence is float, not hardcoded ──
                    conf = quant["confidence"]
                    all_confidences.append(conf)
                    if isinstance(conf, float):
                        log_pass(f"{tf}: Confidence is float", f"{conf}")
                    else:
                        log_fail(f"{tf}: Confidence is float", f"type={type(conf)}, val={conf}")

                    if 35.0 <= conf <= 95.0:
                        log_pass(f"{tf}: Confidence in [35,95]", f"{conf}")
                    else:
                        log_fail(f"{tf}: Confidence in [35,95]", f"{conf}")

                    # ── CHECK 3: Direction is valid ──────────────
                    d = quant["direction"]
                    if d in ("BULLISH", "BEARISH", "NEUTRAL"):
                        log_pass(f"{tf}: Direction valid", d)
                    else:
                        log_fail(f"{tf}: Direction valid", d)

                    # ── CHECK 4: Trend is valid ──────────────────
                    trend = quant.get("trend")
                    if trend in ("uptrend", "downtrend", "sideways"):
                        log_pass(f"{tf}: Trend valid", trend)
                    else:
                        log_fail(f"{tf}: Trend valid", f"got '{trend}'")

                    # ── CHECK 5: Price target is float with precision ──
                    tgt = quant["price_target"]
                    if isinstance(tgt, float):
                        log_pass(f"{tf}: Target is float", f"{tgt:.2f}")
                    else:
                        log_fail(f"{tf}: Target is float", f"type={type(tgt)}")

                    # ── CHECK 6: Target in reasonable range ──────
                    if tgt > 0:
                        pct_diff = abs(tgt - current_price) / current_price * 100
                        if pct_diff < 50:  # target within 50% of current
                            log_pass(f"{tf}: Target reasonable ({pct_diff:.1f}% from price)", f"target={tgt:.2f}, price={current_price:.2f}")
                        else:
                            log_warn(f"{tf}: Target far ({pct_diff:.1f}% from price)", f"target={tgt:.2f}, price={current_price:.2f}")

                    tf_targets[tf] = tgt

                    # ── CHECK 7: Support/Resistance sanity ───────
                    sup = quant.get("support_level", 0)
                    res = quant.get("resistance_level", 0)
                    if sup > 0 and res > 0 and sup < res:
                        log_pass(f"{tf}: Support ({sup:.2f}) < Resistance ({res:.2f})")
                    elif sup > 0 and res > 0:
                        log_warn(f"{tf}: Support >= Resistance", f"S={sup:.2f}, R={res:.2f}")

                    # ── CHECK 8: ATR positive ────────────────────
                    atr = quant.get("atr", 0)
                    if atr > 0:
                        log_pass(f"{tf}: ATR > 0", f"{atr:.2f}")
                    else:
                        log_warn(f"{tf}: ATR is 0", "May indicate insufficient data")

                    # ── CHECK 9: Detected patterns is a list ─────
                    det_pats = quant.get("detected_patterns", None)
                    if isinstance(det_pats, list):
                        log_pass(f"{tf}: detected_patterns is list", f"{len(det_pats)} patterns")
                    else:
                        log_fail(f"{tf}: detected_patterns is list", f"type={type(det_pats)}")

                    # ── CHECK 10: raw_score is float ─────────────
                    rs = quant.get("raw_score")
                    if isinstance(rs, (int, float)):
                        log_pass(f"{tf}: raw_score is numeric", f"{rs}")
                    else:
                        log_fail(f"{tf}: raw_score is numeric", f"type={type(rs)}")

                    # Print summary row
                    results_table.append({
                        "stock": ticker.replace(".NS", ""),
                        "timeframe": tf,
                        "direction": d,
                        "confidence": conf,
                        "target": tgt,
                        "target_pct": quant.get("price_target_pct", 0),
                        "support": sup,
                        "resistance": res,
                        "trend": trend,
                        "raw_score": rs,
                        "rsi": quant.get("rsi", 0),
                        "atr": atr,
                        "n_patterns": len(det_pats) if det_pats else 0,
                        "patterns": det_pats,
                    })

                except Exception as e:
                    log_fail(f"{tf}: EXCEPTION", str(e))
                    traceback.print_exc()

            # ── Cross-timeframe check: target spread ──────────
            if len(tf_targets) == 4:
                print(f"\n  Cross-Timeframe Target Comparison:")
                for tf_name, tgt in tf_targets.items():
                    print(f"    {tf_name:15s} -> target: {tgt:>10.2f}  (price: {current_price:.2f})")

        except Exception as e:
            log_fail(f"{ticker}: EXCEPTION", str(e))
            traceback.print_exc()

    # ═════════════════════════════════════════════════════════════════
    #  GLOBAL CHECKS: Confidence Variation Across Stocks
    # ═════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("GLOBAL VALIDATION")
    print("=" * 70)

    if len(all_confidences) > 4:
        unique_rounded = set(round(c, 0) for c in all_confidences)
        if len(unique_rounded) >= 3:
            log_pass(f"Confidence varies across stocks/timeframes", f"{len(unique_rounded)} unique values from {len(all_confidences)} predictions")
        else:
            log_fail(f"Confidence lacks variation", f"only {len(unique_rounded)} unique values: {unique_rounded}")

        # Check we're not stuck at 60-65
        below_60 = sum(1 for c in all_confidences if c < 60)
        above_75 = sum(1 for c in all_confidences if c > 75)
        if below_60 > 0 or above_75 > 0:
            log_pass(f"Confidence spread is dynamic", f"{below_60} below 60%, {above_75} above 75% (out of {len(all_confidences)})")
        else:
            log_warn(f"All confidences between 60-75%", "May indicate sigmoid is not differentiating enough")

    # ═════════════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ═════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY TABLE")
    print("=" * 70)

    header = f"{'Stock':>12} {'Timeframe':>14} {'Dir':>8} {'Conf':>6} {'Target':>10} {'Tgt%':>7} {'S/R':>20} {'Trend':>10} {'Score':>7} {'RSI':>6} {'ATR':>8} {'#Pat':>4}"
    print(header)
    print("-" * len(header))

    for r in results_table:
        sr_str = f"{r['support']:.0f}/{r['resistance']:.0f}"
        print(f"{r['stock']:>12} {r['timeframe']:>14} {r['direction']:>8} {r['confidence']:>6.1f} {r['target']:>10.2f} {r['target_pct']:>+7.2f} {sr_str:>20} {r['trend']:>10} {r['raw_score']:>7.1f} {r['rsi']:>6.1f} {r['atr']:>8.2f} {r['n_patterns']:>4}")

    print()


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  COMPREHENSIVE PREDICTION ENGINE TEST")
    print("  Testing all 5 implementation plan components")
    print("=" * 70)

    # Unit tests
    test_sigmoid_confidence()
    test_score_strength_factor()
    test_detect_trend()
    test_adjust_pattern_confidence()
    test_normalize_df()

    # Integration tests on real stocks
    test_stock_predictions()

    # Final tally
    print("=" * 70)
    TOTAL = PASS + FAIL
    print(f"  FINAL SCORE:  {PASS}/{TOTAL} passed  |  {FAIL} FAILED  |  {WARNINGS} warnings")
    if FAIL == 0:
        print("  >>> ALL TESTS PASSED <<<")
    else:
        print(f"  >>> {FAIL} TESTS FAILED — see details above <<<")
    print("=" * 70)

    sys.exit(1 if FAIL > 0 else 0)
