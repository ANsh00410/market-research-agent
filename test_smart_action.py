"""
Smart Action Advisor — Accuracy & Sanity Test
================================================
Tests 10 stocks with real market data to verify:
  1. Recommendations are differentiated (not all same action)
  2. BUY MORE aligns with oversold/undervalued conditions
  3. EXIT aligns with overbought/overvalued conditions
  4. Opportunity analysis values are mathematically correct
  5. Target exit prices are sensible
  6. Score varies meaningfully across stocks
  7. Reasons are populated and relevant
"""

import sys
import json
import traceback

import yfinance as yf
import numpy as np
import pandas as pd

# Import the function directly from app.py
sys.path.insert(0, ".")
from app import compute_smart_action

STOCKS = [
    ("Ather Energy Ltd", "ATHERENERG.NS", 20, 696.0),
    ("Reliance Industries", "RELIANCE.NS", 1, 100.0),    # placeholder buy
    ("TCS", "TCS.NS", 5, 3200.0),                         # bought high
    ("HDFC Bank", "HDFCBANK.NS", 10, 750.0),              # near current
    ("Infosys", "INFY.NS", 1, 100.0),                     # placeholder buy
    ("ITC", "ITC.NS", 50, 280.0),                         # near current
    ("SBI", "SBIN.NS", 10, 600.0),                        # bought low
    ("Bharti Airtel", "BHARTIARTL.NS", 5, 1500.0),        # bought low
    ("Sun Pharma", "SUNPHARMA.NS", 8, 1800.0),            # near current
    ("Adani Enterprises", "ADANIENT.NS", 1, 2203.7),      # real buy price
]

PASS = 0
FAIL = 0
WARN = 0

def log_pass(test, detail=""):
    global PASS; PASS += 1
    print(f"  [PASS] {test}" + (f" -- {detail}" if detail else ""))

def log_fail(test, detail=""):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {test}" + (f" -- {detail}" if detail else ""))

def log_warn(test, detail=""):
    global WARN; WARN += 1
    print(f"  [WARN] {test}" + (f" -- {detail}" if detail else ""))


def build_snap(ticker):
    """Build a snap dict from yfinance data (mimics app.py logic)."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    if df.empty:
        return None, None

    df = df.dropna(subset=["Close"])
    if len(df) < 50:
        return None, None

    close = df["Close"]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    w52_high = float(df["High"].tail(252).max())
    w52_low = float(df["Low"].tail(252).min())
    return_1m = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) >= 21 else 0

    vol_series = df["Volume"].replace(0, np.nan).dropna()
    avg_vol = int(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else 0
    today_vol = int(df["Volume"].iloc[-1])

    snap = {
        "current_price": round(price, 2),
        "prev_close": round(prev, 2),
        "today_change": round((price - prev) / prev * 100, 2),
        "rsi": round(rsi, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "week52_high": round(w52_high, 2),
        "week52_low": round(w52_low, 2),
        "return_1m": round(return_1m, 2),
        "avg_volume": avg_vol,
        "today_volume": today_vol,
        "signal": "BULLISH" if price > ma20 and price > ma50 else ("BEARISH" if price < ma20 and price < ma50 else "NEUTRAL"),
    }
    return snap, df


def run_test():
    print("=" * 90)
    print("  SMART ACTION ADVISOR — ACCURACY & SANITY TEST")
    print("  Testing 10 stocks with real market data")
    print("=" * 90)

    results = []
    all_actions = []
    all_scores = []

    for company, ticker, qty, avg_buy in STOCKS:
        print(f"\n{'═' * 90}")
        print(f"  {company} ({ticker}) — Qty: {qty}, Avg Buy: ₹{avg_buy:,.2f}")
        print(f"{'═' * 90}")

        try:
            snap, df = build_snap(ticker)
            if snap is None:
                log_warn(f"No data for {ticker}", "skipping")
                continue

            price = snap["current_price"]
            entry = {"avg_price": avg_buy, "quantity": qty, "id": ticker, "company": company, "ticker": ticker}

            # Run Smart Action Advisor
            smart = compute_smart_action(entry, snap, pred_result=None)

            action = smart["action"]
            score = smart["action_score"]
            pnl_pct = smart["pnl_pct"]
            exit_target = smart["exit_target"]
            w52_pos = smart["w52_position"]
            hyp_gain = smart["if_bought_now_gain"]
            hyp_rupees = smart["if_bought_now_rupees"]
            hyp_qty = smart["hypothetical_qty"]

            all_actions.append(action)
            all_scores.append(score)

            print(f"\n  Price: ₹{price:,.2f}  |  P&L: {pnl_pct:+.2f}%  |  RSI: {snap['rsi']:.1f}")
            print(f"  MA20: ₹{snap['ma20']:,.2f}  |  MA50: ₹{snap['ma50']:,.2f}")
            print(f"  52W: ₹{snap['week52_low']:,.2f} – ₹{snap['week52_high']:,.2f}  ({w52_pos:.1f}% position)")
            print(f"  1M Return: {snap['return_1m']:+.2f}%")
            print(f"\n  >>> ACTION: {smart['action_icon']} {action}  (Score: {score:+.0f})")
            print(f"  >>> Target Exit: ₹{exit_target:,.2f}")

            # Print reasons
            print(f"\n  Reasons:")
            for r in smart["reasons_buy"]:
                print(f"    ▲ {r}")
            for r in smart["reasons_exit"]:
                print(f"    ▼ {r}")
            for r in smart["reasons_hold"]:
                print(f"    ● {r}")

            print(f"\n  Opportunity:")
            print(f"    If invest ₹{smart['hypothetical_invest']:,.0f} more → ~{hyp_qty} shares")
            if hyp_gain > 0:
                print(f"    Potential gain at target: ₹{hyp_rupees:,.0f} (+{hyp_gain:.2f}%)")
            else:
                print(f"    Target below current price — not a good entry")

            # ── Sanity checks ──

            # 1. Action is valid
            if action in ("BUY MORE", "HOLD", "EXIT / BOOK PROFIT"):
                log_pass(f"Action valid", f"{action}")
            else:
                log_fail(f"Action invalid", f"'{action}'")

            # 2. Score is a float
            if isinstance(score, float):
                log_pass(f"Score is float", f"{score:+.1f}")
            else:
                log_fail(f"Score type wrong", f"{type(score)}")

            # 3. If RSI < 35 and price near 52W low, should lean BUY MORE
            if snap["rsi"] < 35 and w52_pos < 30:
                if action == "BUY MORE":
                    log_pass(f"Oversold + near 52W low → BUY MORE", f"RSI={snap['rsi']:.1f}, 52W pos={w52_pos:.0f}%")
                else:
                    log_warn(f"Oversold + near 52W low but not BUY MORE", f"Got {action}, score={score:+.0f}")

            # 4. If RSI > 70 and near 52W high, should lean EXIT
            if snap["rsi"] > 70 and w52_pos > 80:
                if action == "EXIT / BOOK PROFIT":
                    log_pass(f"Overbought + near 52W high → EXIT", f"RSI={snap['rsi']:.1f}, 52W pos={w52_pos:.0f}%")
                else:
                    log_warn(f"Overbought + near 52W high but not EXIT", f"Got {action}, score={score:+.0f}")

            # 5. Exit target must be positive
            if exit_target > 0:
                log_pass(f"Exit target positive", f"₹{exit_target:,.2f}")
            else:
                log_fail(f"Exit target invalid", f"₹{exit_target}")

            # 6. Hypothetical qty calculation correct
            expected_qty = int(10000 / price) if price > 0 else 0
            if hyp_qty == expected_qty:
                log_pass(f"Hypothetical qty correct", f"{hyp_qty} shares for ₹10,000")
            else:
                log_fail(f"Hypothetical qty wrong", f"got {hyp_qty}, expected {expected_qty}")

            # 7. P&L percentage correct
            expected_pnl = (price - avg_buy) / avg_buy * 100.0
            if abs(pnl_pct - expected_pnl) < 0.01:
                log_pass(f"P&L% correct", f"{pnl_pct:+.2f}%")
            else:
                log_fail(f"P&L% wrong", f"got {pnl_pct:+.2f}%, expected {expected_pnl:+.2f}%")

            # 8. Reasons are populated
            total_reasons = len(smart["reasons_buy"]) + len(smart["reasons_exit"]) + len(smart["reasons_hold"])
            if total_reasons >= 2:
                log_pass(f"Reasons populated", f"{total_reasons} reasons given")
            else:
                log_warn(f"Few reasons", f"only {total_reasons}")

            # 9. 52W position in valid range
            if 0 <= w52_pos <= 100:
                log_pass(f"52W position in range", f"{w52_pos:.1f}%")
            else:
                log_fail(f"52W position out of range", f"{w52_pos:.1f}%")

            # 10. If BUY MORE, exit target should be above current price
            if action == "BUY MORE" and exit_target > price:
                log_pass(f"BUY MORE target above price", f"₹{exit_target:,.2f} > ₹{price:,.2f}")
            elif action == "BUY MORE" and exit_target <= price:
                log_fail(f"BUY MORE but target below price!", f"₹{exit_target:,.2f} <= ₹{price:,.2f}")

            results.append({
                "stock": ticker.replace(".NS", ""),
                "price": price,
                "avg_buy": avg_buy,
                "pnl_pct": pnl_pct,
                "rsi": snap["rsi"],
                "w52_pos": w52_pos,
                "action": action,
                "score": score,
                "exit_target": exit_target,
                "hyp_gain": hyp_gain,
                "n_reasons": total_reasons,
            })

        except Exception as e:
            log_fail(f"{ticker}: EXCEPTION", str(e))
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    #  GLOBAL CHECKS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  GLOBAL VALIDATION")
    print(f"{'═' * 90}")

    # Actions should vary across stocks
    unique_actions = set(all_actions)
    if len(unique_actions) >= 2:
        log_pass(f"Actions are differentiated", f"{len(unique_actions)} unique: {unique_actions}")
    else:
        log_warn(f"All actions are the same", f"{unique_actions}")

    # Scores should vary
    unique_scores = set(round(s, 0) for s in all_scores)
    if len(unique_scores) >= 3:
        log_pass(f"Scores vary across stocks", f"{len(unique_scores)} unique rounded scores")
    else:
        log_warn(f"Scores lack variation", f"only {len(unique_scores)} unique")

    # Score range
    if len(all_scores) > 0:
        score_range = max(all_scores) - min(all_scores)
        log_pass(f"Score range", f"min={min(all_scores):+.0f}, max={max(all_scores):+.0f}, range={score_range:.0f}")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 90}")
    print(f"  SUMMARY TABLE")
    print(f"{'═' * 90}")

    header = f"{'Stock':>14} {'Price':>10} {'AvgBuy':>10} {'P&L%':>8} {'RSI':>6} {'52W%':>6} {'Action':>22} {'Score':>7} {'ExitTgt':>10} {'HypGain%':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['stock']:>14} {r['price']:>10,.2f} {r['avg_buy']:>10,.2f} {r['pnl_pct']:>+8.2f} {r['rsi']:>6.1f} {r['w52_pos']:>6.1f} {r['action']:>22} {r['score']:>+7.0f} {r['exit_target']:>10,.2f} {r['hyp_gain']:>+9.2f}")

    # Final tally
    TOTAL = PASS + FAIL
    print(f"\n{'═' * 90}")
    print(f"  FINAL SCORE:  {PASS}/{TOTAL} passed  |  {FAIL} FAILED  |  {WARN} warnings")
    if FAIL == 0:
        print(f"  >>> ALL TESTS PASSED <<<")
    else:
        print(f"  >>> {FAIL} TESTS FAILED — see details above <<<")
    print(f"{'═' * 90}")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    run_test()
