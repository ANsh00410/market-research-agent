# stock_tools.py
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from textblob import TextBlob
from duckduckgo_search import DDGS
from datetime import datetime, timedelta


def get_stock_analysis(company_name: str, ticker: str = None) -> str:
    """
    Fetch stock data and technical analysis for Indian stocks.
    Ticker format for NSE: 'RELIANCE.NS', 'TCS.NS', 'TATAMOTORS.NS'
    Ticker format for BSE: 'RELIANCE.BO'
    """
    try:
        # If no ticker provided, try to guess it
        if not ticker:
            # Common Indian stock mapping
            nse_ticker = company_name.upper().replace(" ", "") + ".NS"
            stock = yf.Ticker(nse_ticker)
        else:
            stock = yf.Ticker(ticker)

        # Get 1 year of historical data
        df = stock.history(period="6mo")

        if df.empty:
            return f"No stock data found for {company_name}. It may not be listed or ticker is wrong."

        # ---- Basic Info ----
        info = stock.info
        current_price = df["Close"].iloc[-1]
        price_1m_ago = df["Close"].iloc[-22] if len(df) > 22 else df["Close"].iloc[0]
        price_3m_ago = df["Close"].iloc[-66] if len(df) > 66 else df["Close"].iloc[0]
        price_1y_ago = df["Close"].iloc[0]

        monthly_return = ((current_price - price_1m_ago) / price_1m_ago) * 100
        quarterly_return = ((current_price - price_3m_ago) / price_3m_ago) * 100
        yearly_return = ((current_price - price_1y_ago) / price_1y_ago) * 100

        # ---- Technical Indicators ----
        # Moving Averages
        df["MA20"] = ta.sma(df["Close"], length=20)
        df["MA50"] = ta.sma(df["Close"], length=50)
        df["MA200"] = ta.sma(df["Close"], length=200)

        # RSI (Relative Strength Index)
        df["RSI"] = ta.rsi(df["Close"], length=14)

        # MACD
        macd = ta.macd(df["Close"])
        df = pd.concat([df, macd], axis=1)

        # Bollinger Bands
        bbands = ta.bbands(df["Close"], length=20)
        df = pd.concat([df, bbands], axis=1)

        # Get latest values
        latest = df.iloc[-1]
        rsi = latest["RSI"]
        ma20 = latest["MA20"]
        ma50 = latest["MA50"]
        ma200 = latest["MA200"]

        # ---- Interpret Signals ----
        signals = []

        # RSI Signal
        if rsi < 30:
            rsi_signal = "🟢 OVERSOLD (potential buy opportunity)"
        elif rsi > 70:
            rsi_signal = "🔴 OVERBOUGHT (potential correction ahead)"
        else:
            rsi_signal = "🟡 NEUTRAL"

        # Moving Average Trend
        if current_price > ma20 > ma50:
            ma_signal = "🟢 BULLISH (price above short & medium term averages)"
        elif current_price < ma20 < ma50:
            ma_signal = "🔴 BEARISH (price below short & medium term averages)"
        else:
            ma_signal = "🟡 MIXED signals"

        # 52-week high/low
        week52_high = df["High"].max()
        week52_low = df["Low"].min()
        distance_from_high = ((week52_high - current_price) / week52_high) * 100

        # Volume trend
        avg_volume = df["Volume"].tail(20).mean()
        current_volume = df["Volume"].iloc[-1]
        volume_signal = (
            "📈 Above average volume"
            if current_volume > avg_volume
            else "📉 Below average volume"
        )

        # ---- Format Report ----
        report = f"""
📈 STOCK ANALYSIS: {company_name}
{'='*50}

💰 PRICE DATA:
  Current Price: ₹{current_price:.2f}
  52-Week High:  ₹{week52_high:.2f}
  52-Week Low:   ₹{week52_low:.2f}
  Distance from 52W High: {distance_from_high:.1f}%

📊 RETURNS:
  1 Month:  {monthly_return:+.2f}%
  3 Months: {quarterly_return:+.2f}%
  1 Year:   {yearly_return:+.2f}%

🔧 TECHNICAL INDICATORS:
  RSI (14):  {rsi:.1f} → {rsi_signal}
  MA Signal: {ma_signal}
  MA20: ₹{ma20:.2f} | MA50: ₹{ma50:.2f} | MA200: ₹{ma200:.2f}
  {volume_signal}

📌 OVERALL TECHNICAL VIEW:
"""
        # Simple overall view
        bullish_signals = sum(
            [
                current_price > ma20,
                current_price > ma50,
                current_price > ma200,
                rsi < 60,
                current_price > (week52_low * 1.1),
            ]
        )

        if bullish_signals >= 4:
            report += "  🟢 MOSTLY BULLISH — Stock shows positive technical setup\n"
        elif bullish_signals <= 2:
            report += "  🔴 MOSTLY BEARISH — Stock shows weak technical setup\n"
        else:
            report += "  🟡 NEUTRAL — Mixed signals, wait for clearer trend\n"

        report += (
            "\n⚠️ Disclaimer: This is technical analysis only, not financial advice."
        )

        return report

    except Exception as e:
        return f"Error analyzing stock: {str(e)}"


def get_stock_sentiment(company_name: str) -> str:
    """Analyze news sentiment for a company's stock."""
    try:
        with DDGS() as ddgs:
            news_results = list(
                ddgs.news(f"{company_name} stock NSE India", max_results=10)
            )

        if not news_results:
            return "No recent news found for sentiment analysis."

        sentiments = []
        news_summary = []

        for article in news_results:
            title = article.get("title", "")
            body = article.get("body", "")
            text = title + " " + body

            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to +1
            sentiments.append(polarity)

            sentiment_label = (
                "🟢 Positive"
                if polarity > 0.1
                else ("🔴 Negative" if polarity < -0.1 else "🟡 Neutral")
            )
            news_summary.append(f"  {sentiment_label} | {title[:80]}...")

        avg_sentiment = sum(sentiments) / len(sentiments)

        if avg_sentiment > 0.1:
            overall = "🟢 POSITIVE — Recent news sentiment is favorable"
        elif avg_sentiment < -0.1:
            overall = "🔴 NEGATIVE — Recent news sentiment is unfavorable"
        else:
            overall = "🟡 NEUTRAL — Mixed news sentiment"

        result = f"""
📰 NEWS SENTIMENT ANALYSIS: {company_name}
{'='*50}
Overall Sentiment Score: {avg_sentiment:.3f} (-1 worst, +1 best)
Overall View: {overall}

Recent Headlines:
""" + "\n".join(
            news_summary[:7]
        )

        return result

    except Exception as e:
        return f"Sentiment error: {str(e)}"
