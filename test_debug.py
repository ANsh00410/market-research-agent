import yfinance as yf
import pandas as pd
from prediction_engine import run_full_prediction
import json

def test_debug():
    ticker = "RELIANCE.NS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    snap = {
        "current_price": float(df["Close"].iloc[-1]),
        "today_change": 0.5,
        "rsi": 40.0,
        "ma20": float(df["Close"].rolling(20).mean().iloc[-1]),
        "ma50": float(df["Close"].rolling(50).mean().iloc[-1]),
        "week52_high": float(df["High"].max()),
        "week52_low": float(df["Low"].min()),
        "return_1m": 2.0,
        "avg_volume": float(df["Volume"].mean()),
        "today_volume": float(df["Volume"].iloc[-1]),
    }
    
    print(f"Testing {ticker} on Long Term...")
    result = run_full_prediction("Reliance", ticker, snap, df, term="Long Term")
    print(f"Quant:\n{json.dumps(result['quant_prediction'], indent=2)}")
    print(f"\nSentiment:\n{result['sentiment_report']}")
    print(f"\nOrchestrator:\n{json.dumps(result['prediction'], indent=2)}")

if __name__ == "__main__":
    test_debug()
