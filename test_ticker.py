import yfinance as yf

def fetch_cnbc_ticker_data():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "LARSEN.NS", 
        "KOTAKBANK.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", 
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TATASTEEL.NS",
        "M&M.NS", "NTPC.NS", "POWERGRID.NS", "INDUSINDBK.NS", "NESTLEIND.NS"
    ]
    try:
        df = yf.download(tickers, period="5d", progress=False)
        if df.empty:
            return []
            
        closes = df['Close']
        volumes = df['Volume']
        print("Closes columns:", closes.columns)
        
        results = []
        for t in tickers:
            if t in closes.columns:
                c_vals = closes[t].dropna()
                v_vals = volumes[t].dropna() if t in volumes.columns else []
                if len(c_vals) >= 2:
                    c0, c1 = float(c_vals.iloc[-2]), float(c_vals.iloc[-1])
                    vol = float(v_vals.iloc[-1]) if len(v_vals) else 0
                    chg = c1 - c0
                    
                    if vol >= 1_000_000:
                        vol_str = f"{vol/1_000_000:.1f}m"
                    elif vol >= 1_000:
                        vol_str = f"{vol/1_000:.1f}k"
                    else:
                        vol_str = str(int(vol))
                        
                    results.append({
                        "name": t.replace(".NS", ""),
                        "price": c1,
                        "chg": chg,
                        "vol": vol_str
                    })
        return results
    except Exception as e:
        print("Error:", e)
        return []

if __name__ == "__main__":
    res = fetch_cnbc_ticker_data()
    print("Results:", len(res))

