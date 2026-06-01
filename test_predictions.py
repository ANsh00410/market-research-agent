import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import random
from prediction_engine import run_full_prediction

def test_predictions():
    stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    
    test_dates = ["2024-01-05"]
    timeframes = ["Swing Trade", "Short Term", "Medium Term", "Long Term"]
    
    results = []
    
    for ticker in stocks:
        stock = yf.Ticker(ticker)
        df = stock.history(start="2020-01-01", end="2024-12-31")
        if df.empty:
            continue
            
        for test_date_str in test_dates:
            try:
                test_date = pd.to_datetime(test_date_str).tz_localize(df.index.tz)
                
                past_df = df[df.index <= test_date].copy()
                if len(past_df) < 50:
                    continue
                    
                delta = past_df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                actual_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

                snap = {
                    "current_price": float(past_df["Close"].iloc[-1]),
                    "today_change": float((past_df["Close"].iloc[-1] - past_df["Close"].iloc[-2]) / past_df["Close"].iloc[-2] * 100),
                    "rsi": actual_rsi,
                    "ma20": float(past_df["Close"].rolling(20).mean().iloc[-1]),
                    "ma50": float(past_df["Close"].rolling(50).mean().iloc[-1]),
                    "week52_high": float(past_df["High"].tail(252).max()),
                    "week52_low": float(past_df["Low"].tail(252).min()),
                    "return_1m": float((past_df["Close"].iloc[-1] - past_df["Close"].iloc[-20]) / past_df["Close"].iloc[-20] * 100) if len(past_df) >= 20 else 0,
                    "avg_volume": int(past_df["Volume"].tail(20).mean()),
                    "today_volume": int(past_df["Volume"].iloc[-1]),
                }
                
                for tf in timeframes:
                    print(f"Testing {ticker} on {test_date_str} for {tf}...")
                    
                    pred_result = run_full_prediction(ticker, ticker, snap, past_df, term=tf)
                    prediction = pred_result["prediction"]
                    
                    predicted_dir = prediction.get("direction", "NEUTRAL")
                    target = prediction.get("price_target")
                    
                    results.append({
                        "ticker": ticker,
                        "date": test_date_str,
                        "timeframe": tf,
                        "predicted_dir": predicted_dir,
                        "target": target,
                        "confidence": prediction.get("confidence")
                    })
            except Exception as e:
                print(f"Error on {ticker} {test_date_str}: {e}")

    print("\n--- RESULTS ---")
    for r in results:
        print(f"{r['ticker']} | {r['timeframe']}: Pred={r['predicted_dir']}, Target={r['target']}, Conf={r['confidence']}")

if __name__ == "__main__":
    test_predictions()
