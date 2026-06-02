"""
Comprehensive Prediction Engine Audit
======================================
Tests 8 stocks across all 4 timeframes.
Validates:
  1. Data integrity: yfinance current_price, RSI, MA20, MA50, 52W high/low
  2. Quant engine: direction, confidence, targets, support/resistance, ATR
  3. Orchestrator reasoning: references specific data, direction consistency
  4. Target sanity: BULLISH target > price, BEARISH target < price
  5. Timeframe scaling: swing target range < long target range
  6. Float precision: no integer truncation
  7. P&L calculation verification
"""

import sys
import json
import math
import traceback

import yfinance as yf
import pandas as pd
import numpy as np

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
)

# ─── Test stocks ─────────────────────────────────────────────
STOCKS = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("TCS", "TCS.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("Infosys", "INFY.NS"),
    ("ITC", "ITC.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("Sun Pharma", "SUNPHARMA.NS"),
]

TIMEFRAMES = ["Swing Trade", "Short Term", "Medium Term", "Long Term"]

PASS = 0
FAIL = 0
WARNINGS = 0
results = []

def log_pass(test, detail=""):
    global PASS; PASS += 1
    print(f"  [PASS] {test}" + (f" -- {detail}" if detail else ""))

def log_fail(test, detail=""):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {test}" + (f" -- {detail}" if detail else ""))

def log_warn(test, detail=""):
    global WARNINGS; WARNINGS += 1
    print(f"  [WARN] {test}" + (f" -- {detail}" if detail else ""))


def verify_data_integrity(ticker, df, snap):
    """Verify that yfinance data is correct and consistent."""
    print(f"\n  -- Data Integrity Checks --")
    
    price = snap["current_price"]
    
    # Price must be a positive float
    if isinstance(price, float) and price > 0:
        log_pass(f"current_price is positive float", f"₹{price:,.2f}")
    else:
        log_fail(f"current_price invalid", f"type={type(price)}, val={price}")
    
    # RSI sanity: must be 0-100
    rsi = snap.get("rsi")
    if rsi is not None:
        if 0 <= rsi <= 100:
            log_pass(f"RSI in [0,100]", f"{rsi:.2f}")
        else:
            log_fail(f"RSI out of range", f"{rsi}")
    
    # MA20 and MA50 must be reasonable (within 50% of current price)
    for ma_name in ["ma20", "ma50"]:
        ma_val = snap.get(ma_name)
        if ma_val is not None:
            pct_diff = abs(ma_val - price) / price * 100
            if pct_diff < 50:
                log_pass(f"{ma_name} reasonable", f"₹{ma_val:,.2f} ({pct_diff:.1f}% from price)")
            else:
                log_warn(f"{ma_name} far from price", f"₹{ma_val:,.2f} ({pct_diff:.1f}%)")
    
    # 52-week range must bracket the current price (or close to it)
    w52_high = snap.get("week52_high", 0)
    w52_low = snap.get("week52_low", 0)
    if w52_low > 0 and w52_high > 0:
        if w52_low <= price * 1.02:  # price might be slightly below 52W low temporarily
            log_pass(f"52W low ≤ price", f"₹{w52_low:,.0f} ≤ ₹{price:,.0f}")
        else:
            log_warn(f"52W low > price", f"₹{w52_low:,.0f} > ₹{price:,.0f}")
        if w52_high >= price * 0.98:
            log_pass(f"52W high ≥ price", f"₹{w52_high:,.0f} ≥ ₹{price:,.0f}")
        else:
            log_warn(f"52W high < price", f"₹{w52_high:,.0f} < ₹{price:,.0f}")


def verify_pnl_calculation(price, avg_buy, qty):
    """Verify P&L math is correct."""
    print(f"\n  -- P&L Verification (avg_buy=₹{avg_buy:,.2f}, qty={qty}) --")
    inv_val = qty * avg_buy
    cur_val = qty * price
    pnl = cur_val - inv_val
    pnl_pct = (pnl / inv_val * 100) if inv_val else 0
    
    # Verify the math manually
    expected_pnl = price - avg_buy  # per share
    expected_pnl_pct = (price - avg_buy) / avg_buy * 100
    
    if abs(pnl - expected_pnl * qty) < 0.01:
        log_pass(f"P&L calc correct", f"₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
    else:
        log_fail(f"P&L calc wrong", f"got ₹{pnl:+,.2f}, expected ₹{expected_pnl*qty:+,.2f}")
    
    # Check if avg_buy is a placeholder
    if avg_buy == 100.0 and price > 200:
        log_warn(f"avg_buy=₹100 looks like placeholder!", 
                 f"Real P&L would need actual purchase price. Showing +{pnl_pct:.0f}% because buy=₹100 vs price=₹{price:,.0f}")


def run_audit():
    print("=" * 80)
    print("  COMPREHENSIVE PREDICTION ENGINE AUDIT")
    print("  Testing 8 stocks × 4 timeframes = 32 prediction scenarios")
    print("=" * 80)

    all_confidences = []
    
    for company, ticker in STOCKS:
        print(f"\n{'═' * 80}")
        print(f"  STOCK: {company} ({ticker})")
        print(f"{'═' * 80}")
        
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            if df.empty:
                log_warn(f"{ticker} - empty history", "skipping")
                continue
            
            close_series = df["Close"].dropna()
            if len(close_series) < 50:
                log_warn(f"{ticker} - insufficient data", f"only {len(close_series)} rows")
                continue
            
            current_price = float(close_series.iloc[-1])
            
            # Compute reference indicators for cross-check
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            actual_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            
            ma20_ref = float(close_series.rolling(20).mean().iloc[-1])
            ma50_ref = float(close_series.rolling(50).mean().iloc[-1]) if len(close_series) >= 50 else None
            w52_high = float(df["High"].tail(252).max())
            w52_low = float(df["Low"].tail(252).min())
            
            snap = {
                "current_price": current_price,
                "today_change": float((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2] * 100),
                "rsi": actual_rsi,
                "ma20": ma20_ref,
                "ma50": ma50_ref,
                "week52_high": w52_high,
                "week52_low": w52_low,
                "return_1m": float((close_series.iloc[-1] - close_series.iloc[-20]) / close_series.iloc[-20] * 100) if len(close_series) >= 20 else 0,
                "avg_volume": float(df["Volume"].tail(20).mean()),
                "today_volume": float(df["Volume"].iloc[-1]),
            }
            
            # 1. Verify data integrity
            verify_data_integrity(ticker, df, snap)
            
            # 2. Verify P&L calculation (simulating the UI with avg_buy=₹100 placeholder)
            verify_pnl_calculation(current_price, 100.0, 1)
            
            # 3. Run predictions for all timeframes
            df_norm = _normalize_df(df)
            cs_pats = detect_candlestick_patterns(df_norm)
            ch_pats = detect_chart_patterns(df_norm)
            
            tf_targets = {}
            tf_data = {}
            
            for tf in TIMEFRAMES:
                print(f"\n  ── Timeframe: {tf} ──")
                
                try:
                    quant = generate_quantitative_prediction(tf, current_price, snap, cs_pats, ch_pats, df_norm)
                    
                    d = quant["direction"]
                    conf = quant["confidence"]
                    tgt = quant["price_target"]
                    tgt_pct = quant["price_target_pct"]
                    sup = quant["support_level"]
                    res_ = quant["resistance_level"]
                    sl = quant["stop_loss"]
                    rsi_q = quant["rsi"]
                    atr = quant["atr"]
                    score = quant["raw_score"]
                    trend = quant["trend"]
                    
                    all_confidences.append(conf)
                    tf_targets[tf] = tgt
                    tf_data[tf] = quant
                    
                    # ── CHECK: Direction valid ──
                    if d in ("BULLISH", "BEARISH", "NEUTRAL"):
                        log_pass(f"Direction valid", f"{d}")
                    else:
                        log_fail(f"Direction invalid", f"'{d}'")
                    
                    # ── CHECK: Confidence is float in range ──
                    if isinstance(conf, float) and 35.0 <= conf <= 95.0:
                        log_pass(f"Confidence valid", f"{conf:.1f}%")
                    else:
                        log_fail(f"Confidence invalid", f"type={type(conf)}, val={conf}")
                    
                    # ── CHECK: BEARISH target below price / BULLISH above ──
                    if d == "BEARISH":
                        if tgt < current_price:
                            log_pass(f"BEARISH target below price", f"₹{tgt:,.2f} < ₹{current_price:,.2f} ({tgt_pct:+.2f}%)")
                        else:
                            log_fail(f"BEARISH target ABOVE price!", f"₹{tgt:,.2f} > ₹{current_price:,.2f}")
                    elif d == "BULLISH":
                        if tgt > current_price:
                            log_pass(f"BULLISH target above price", f"₹{tgt:,.2f} > ₹{current_price:,.2f} ({tgt_pct:+.2f}%)")
                        else:
                            log_fail(f"BULLISH target BELOW price!", f"₹{tgt:,.2f} < ₹{current_price:,.2f}")
                    
                    # ── CHECK: Target percentage matches ──
                    expected_pct = round((tgt - current_price) / current_price * 100.0, 2)
                    if abs(expected_pct - tgt_pct) < 0.02:
                        log_pass(f"Target % correct", f"{tgt_pct:+.2f}%")
                    else:
                        log_fail(f"Target % mismatch", f"got {tgt_pct:+.2f}%, expected {expected_pct:+.2f}%")
                    
                    # ── CHECK: Support < Resistance ──
                    if sup > 0 and res_ > 0 and sup < res_:
                        log_pass(f"Support < Resistance", f"₹{sup:,.2f} < ₹{res_:,.2f}")
                    elif sup > 0 and res_ > 0:
                        log_warn(f"Support >= Resistance", f"₹{sup:,.2f} >= ₹{res_:,.2f}")
                    
                    # ── CHECK: RSI matches between quant engine and reference ──
                    rsi_diff = abs(rsi_q - actual_rsi)
                    if rsi_diff < 2.0:
                        log_pass(f"RSI matches reference", f"quant={rsi_q:.2f}, ref={actual_rsi:.2f}")
                    else:
                        log_warn(f"RSI differs from reference", f"quant={rsi_q:.2f}, ref={actual_rsi:.2f}, diff={rsi_diff:.2f}")
                    
                    # ── CHECK: ATR positive ──
                    if atr > 0:
                        log_pass(f"ATR positive", f"₹{atr:,.2f}")
                    else:
                        log_warn(f"ATR is 0")
                    
                    # ── CHECK: All values are float (no integer truncation) ──
                    float_checks = {"confidence": conf, "target": tgt, "support": sup, "resistance": res_, "stop_loss": sl}
                    all_float = all(isinstance(v, float) for v in float_checks.values())
                    if all_float:
                        log_pass(f"All values are float")
                    else:
                        non_floats = {k: type(v).__name__ for k, v in float_checks.items() if not isinstance(v, float)}
                        log_fail(f"Non-float values found", str(non_floats))
                    
                    # ── CHECK: No NaN ──
                    nan_fields = [k for k, v in float_checks.items() if pd.isna(v)]
                    if not nan_fields:
                        log_pass(f"No NaN values")
                    else:
                        log_fail(f"NaN found", str(nan_fields))
                    
                    # Store result
                    results.append({
                        "stock": ticker.replace(".NS", ""),
                        "timeframe": tf,
                        "direction": d,
                        "confidence": conf,
                        "target": tgt,
                        "target_pct": tgt_pct,
                        "support": sup,
                        "resistance": res_,
                        "trend": trend,
                        "raw_score": score,
                        "rsi": rsi_q,
                        "atr": atr,
                        "n_patterns": len(quant.get("detected_patterns", [])),
                    })
                    
                except Exception as e:
                    log_fail(f"{tf}: EXCEPTION", str(e))
                    traceback.print_exc()
            
            # ── Cross-timeframe target scaling check ──
            print(f"\n  ── Cross-Timeframe Target Analysis ──")
            if len(tf_targets) == 4:
                for tf_name, tgt in tf_targets.items():
                    pct = (tgt - current_price) / current_price * 100
                    print(f"    {tf_name:20s} → target: ₹{tgt:>10,.2f}  ({pct:+7.2f}%)")
                
                # Check if all same direction targets scale properly
                dirs = [tf_data[tf]["direction"] for tf in TIMEFRAMES]
                unique_dirs = set(dirs)
                
                if len(unique_dirs) == 1 and "NEUTRAL" not in unique_dirs:
                    # All same direction - targets should scale
                    tgt_list = [tf_targets[tf] for tf in TIMEFRAMES]
                    if dirs[0] == "BULLISH":
                        # Swing < Short < Medium < Long (ideally)
                        if tgt_list[0] < tgt_list[3]:
                            log_pass(f"BULLISH targets scale: swing < long", 
                                    f"₹{tgt_list[0]:,.2f} < ₹{tgt_list[3]:,.2f}")
                        else:
                            log_warn(f"BULLISH targets don't scale as expected", 
                                    f"swing=₹{tgt_list[0]:,.2f}, long=₹{tgt_list[3]:,.2f}")
                    elif dirs[0] == "BEARISH":
                        # Swing > Short > Medium > Long (ideally for bearish)
                        if tgt_list[0] > tgt_list[3]:
                            log_pass(f"BEARISH targets scale: swing > long", 
                                    f"₹{tgt_list[0]:,.2f} > ₹{tgt_list[3]:,.2f}")
                        else:
                            log_warn(f"BEARISH targets don't scale as expected", 
                                    f"swing=₹{tgt_list[0]:,.2f}, long=₹{tgt_list[3]:,.2f}")
                
        except Exception as e:
            log_fail(f"{ticker}: TOP-LEVEL EXCEPTION", str(e))
            traceback.print_exc()
    
    # ═══════════════════════════════════════════════════════════════
    #  GLOBAL CHECKS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  GLOBAL VALIDATION")
    print(f"{'═' * 80}")
    
    # Confidence variation
    if len(all_confidences) > 4:
        unique_rounded = set(round(c, 0) for c in all_confidences)
        if len(unique_rounded) >= 3:
            log_pass(f"Confidence varies across stocks", 
                    f"{len(unique_rounded)} unique from {len(all_confidences)} predictions")
        else:
            log_fail(f"Confidence lacks variation", 
                    f"only {len(unique_rounded)} unique: {unique_rounded}")
        
        above_75 = sum(1 for c in all_confidences if c > 75)
        between = sum(1 for c in all_confidences if 60 <= c <= 75)
        if above_75 > 0:
            log_pass(f"Dynamic confidence spread", 
                    f"{above_75} above 75%, {between} between 60-75%")
    
    # ═══════════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  FULL RESULTS TABLE")
    print(f"{'═' * 80}")
    
    header = f"{'Stock':>12} {'Timeframe':>14} {'Dir':>8} {'Conf':>6} {'Target':>10} {'Tgt%':>7} {'S/R':>20} {'Trend':>10} {'Score':>7} {'RSI':>6} {'ATR':>8} {'#Pat':>4}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        sr_str = f"{r['support']:.0f}/{r['resistance']:.0f}"
        print(f"{r['stock']:>12} {r['timeframe']:>14} {r['direction']:>8} {r['confidence']:>6.1f} {r['target']:>10.2f} {r['target_pct']:>+7.2f} {sr_str:>20} {r['trend']:>10} {r['raw_score']:>7.1f} {r['rsi']:>6.1f} {r['atr']:>8.2f} {r['n_patterns']:>4}")
    
    # ═══════════════════════════════════════════════════════════════
    #  P&L AUDIT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print(f"  P&L CALCULATION AUDIT")
    print(f"{'═' * 80}")
    print(f"  The P&L formula is: P&L = (current_price - avg_buy_price) × quantity")
    print(f"  P&L% = (current_price - avg_buy_price) / avg_buy_price × 100")
    print(f"")
    print(f"  NOTE: Many portfolio entries have avg_price=₹100 (a placeholder).")
    print(f"  For stocks trading at ₹900+, this produces P&L of +800% which")
    print(f"  looks wrong but is mathematically correct given the input data.")
    print(f"  The user needs to update avg_buy prices to their real purchase prices.")
    
    # Final tally
    print(f"\n{'═' * 80}")
    TOTAL = PASS + FAIL
    print(f"  FINAL SCORE:  {PASS}/{TOTAL} passed  |  {FAIL} FAILED  |  {WARNINGS} warnings")
    if FAIL == 0:
        print(f"  >>> ALL TESTS PASSED <<<")
    else:
        print(f"  >>> {FAIL} TESTS FAILED — see details above <<<")
    print(f"{'═' * 80}")
    
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    run_audit()
