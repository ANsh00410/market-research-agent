"""
app.py — Streamlit UI for Indian Market Research Agent + Portfolio Tracker
With Candlestick Charts + AI Prediction Engine (Bulkowski-based)
Run with: python -m streamlit run app.py
"""

import os
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
import news_engine
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# ─────────────────────────── Page config ───────────────────────────────────

st.set_page_config(
    page_title="Indian Market Research Agent",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── Custom CSS ────────────────────────────────────

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
  h1,h2,h3{font-family:'Syne',sans-serif!important;}

  .hero-header{
    background:linear-gradient(135deg,#FF6B35 0%,#FF9A3C 40%,#FFB347 100%);
    border-radius:16px;padding:36px 40px;margin-bottom:28px;
    color:white;text-align:center;box-shadow:0 8px 32px rgba(255,107,53,.3);
  }
  .hero-header h1{font-family:'Syne',sans-serif!important;font-size:2.4rem;
    font-weight:800;margin:0 0 8px 0;letter-spacing:-0.5px;}
  .hero-header p{font-size:1.05rem;opacity:.92;margin:0;}

  .stButton>button{
    border-radius:24px!important;font-family:'DM Sans',sans-serif!important;
    font-weight:500!important;transition:all .2s ease!important;
    border:1.5px solid #FF6B35!important;color:#FF6B35!important;
    background:transparent!important;
  }
  .stButton>button:hover{
    background:#FF6B35!important;color:white!important;
    transform:translateY(-1px)!important;
    box-shadow:0 4px 12px rgba(255,107,53,.25)!important;
  }
  .status-log{
    background:#0e1117;border:1px solid #2d2d2d;border-radius:10px;
    padding:16px 20px;font-family:monospace;font-size:.82rem;
    color:#00ff88;max-height:200px;overflow-y:auto;line-height:1.8;
  }

  /* ── Prediction cards ── */
  .pred-bull{
    background:linear-gradient(135deg,#0d5e2a 0%,#1a8c42 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #00ff88;
    box-shadow:0 4px 20px rgba(0,255,136,.2);margin-bottom:16px;
  }
  .pred-bear{
    background:linear-gradient(135deg,#5e1a1a 0%,#a02020 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #ff4444;
    box-shadow:0 4px 20px rgba(255,68,68,.2);margin-bottom:16px;
  }
  .pred-neutral{
    background:linear-gradient(135deg,#2a2a0d 0%,#6b6b00 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #ffd700;
    box-shadow:0 4px 20px rgba(255,215,0,.15);margin-bottom:16px;
  }
  .pred-direction{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:1px;}
  .pred-pattern{font-size:.9rem;opacity:.85;margin-top:4px;}
  .pred-target{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;margin-top:10px;}
  .pred-conf{font-size:.82rem;opacity:.8;margin-top:4px;}
  .pred-reasoning{
    font-size:.85rem;opacity:.9;margin-top:12px;
    border-top:1px solid rgba(255,255,255,.2);padding-top:10px;line-height:1.5;
  }
  .pred-metrics{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;}
  .pred-metric{
    background:rgba(255,255,255,.12);border-radius:8px;
    padding:7px 12px;font-size:.8rem;
  }
  .pred-metric-label{opacity:.7;font-size:.72rem;display:block;}
  .conf-bar-bg{width:100%;height:6px;background:rgba(255,255,255,.2);border-radius:3px;margin-top:6px;}
  .conf-bar{height:6px;border-radius:3px;background:#00ff88;}

  /* ── Pattern pills ── */
  .ppill-bull{background:#d4edda;color:#155724;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}
  .ppill-bear{background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}
  .ppill-neutral{background:#fff3cd;color:#856404;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}

  .badge-bull{background:#d4edda;color:#155724;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}
  .badge-bear{background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}
  .badge-neutral{background:#fff3cd;color:#856404;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}

  .section-title{
    font-family:'Syne',sans-serif!important;font-size:1.3rem;
    font-weight:700;color:#1a1a1a;
    border-bottom:2px solid #FF6B35;padding-bottom:8px;margin-bottom:20px;
  }
  .profit{color:#22863a!important;font-weight:600;}
  .loss{color:#cb2431!important;font-weight:600;}

  /* ── Ticker board ── */
  .ticker-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:24px;}
  .tc-green{
    background:linear-gradient(160deg,#0a4d1f 0%,#1a7a35 60%,#0d5e27 100%);
    border:1px solid #00c853;border-radius:12px;padding:16px 18px;
    transition:transform .15s,box-shadow .15s;
  }
  .tc-red{
    background:linear-gradient(160deg,#4d0a0a 0%,#8c1a1a 60%,#5e0d0d 100%);
    border:1px solid #ff1744;border-radius:12px;padding:16px 18px;
    transition:transform .15s,box-shadow .15s;
  }
  .tc-name{font-family:'Syne',sans-serif;font-size:.76rem;font-weight:700;
    color:#FFD700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .tc-ticker-tag{font-size:.67rem;opacity:.55;color:white;font-family:monospace;margin-bottom:4px;}
  .tc-price{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
    color:white;line-height:1.1;margin-bottom:6px;}
  .tc-change-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;}
  .tc-arrow-up{color:#00ff88;font-size:1.3rem;}
  .tc-arrow-down{color:#ff6666;font-size:1.3rem;}
  .tc-chg-val{font-size:.9rem;font-weight:700;color:white;}
  .tc-chg-pct-g{font-size:.82rem;font-weight:700;color:#00ff88;}
  .tc-chg-pct-r{font-size:.82rem;font-weight:700;color:#ff9999;}
  .tc-pnl-row{font-size:.76rem;color:rgba(255,255,255,.8);
    border-top:1px solid rgba(255,255,255,.15);padding-top:6px;margin-top:6px;}
  .board-hdr-g{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    color:#00c853;border-left:4px solid #00c853;padding-left:12px;margin:20px 0 14px;}
  .board-hdr-r{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    color:#ff1744;border-left:4px solid #ff1744;padding-left:12px;margin:20px 0 14px;}

  /* ── NSE Market Watch ── */
  .mw-ticker-row{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 16px; border-radius:8px; margin-bottom:6px;
    border-left:4px solid transparent;
    background:rgba(255,255,255,.03);
    transition:background .15s;
  }
  .mw-ticker-row:hover{background:rgba(255,255,255,.07);}
  .mw-green{border-left-color:#00c853;}
  .mw-red  {border-left-color:#ff1744;}
  .mw-name {font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;color:white;min-width:160px;}
  .mw-sym  {font-family:monospace;font-size:.72rem;color:#aaa;min-width:100px;}
  .mw-price{font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:white;min-width:90px;text-align:right;}
  .mw-chg-g{font-size:.82rem;font-weight:700;color:#00ff88;min-width:80px;text-align:right;}
  .mw-chg-r{font-size:.82rem;font-weight:700;color:#ff6666;min-width:80px;text-align:right;}
  .mw-vol  {font-size:.74rem;color:#888;min-width:90px;text-align:right;}
  .mw-sector{font-size:.7rem;color:#FF9A3C;min-width:130px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .mw-header{
    display:flex;align-items:center;justify-content:space-between;
    padding:8px 16px;border-radius:6px;
    background:rgba(255,107,53,.15);
    font-size:.75rem;font-weight:700;color:#FF9A3C;
    margin-bottom:10px;letter-spacing:.5px;text-transform:uppercase;
  }
  .mw-stat-g{background:rgba(0,200,83,.12);border:1px solid rgba(0,200,83,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-r{background:rgba(255,23,68,.12);border:1px solid rgba(255,23,68,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-n{background:rgba(255,165,0,.1);border:1px solid rgba(255,165,0,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:white;}
  .mw-stat-lbl{font-size:.74rem;color:#aaa;margin-top:2px;}

  /* ── Smart Recommendations ── */
  .rec-card-buy {
    background: linear-gradient(135deg, #0a3d1f 0%, #0d5c2a 100%);
    border: 1px solid #00c853; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 12px;
    border-left: 5px solid #00c853;
    box-shadow: 0 4px 16px rgba(0,200,83,.15);
  }
  .rec-card-sell {
    background: linear-gradient(135deg, #3d0a0a 0%, #5c0d0d 100%);
    border: 1px solid #ff1744; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 12px;
    border-left: 5px solid #ff1744;
    box-shadow: 0 4px 16px rgba(255,23,68,.15);
  }
  .rec-card-neutral {
    background: rgba(255,255,255,.04);
    border: 1px solid #555; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 12px;
    border-left: 5px solid #888;
  }
  .rec-rank {
    font-family:'Syne',sans-serif; font-size:1.6rem;
    font-weight:900; opacity:.35; margin-right:10px;
  }
  .rec-name {
    font-family:'Syne',sans-serif; font-size:1.05rem;
    font-weight:700; color:white;
  }
  .rec-ticker { font-size:.75rem; color:#FFD700; font-family:monospace; }
  .rec-score-bar-bg {
    width:100%; height:8px; background:rgba(255,255,255,.15);
    border-radius:4px; margin:6px 0;
  }
  .rec-score-bar-g { height:8px; border-radius:4px; background:linear-gradient(90deg,#00c853,#69f0ae); }
  .rec-score-bar-r { height:8px; border-radius:4px; background:linear-gradient(90deg,#ff1744,#ff6e40); }
  .rec-price { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:white; }
  .rec-meta  { font-size:.78rem; color:rgba(255,255,255,.7); margin-top:4px; }
  .rec-breakdown { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
  .rec-pill {
    background:rgba(255,255,255,.1); border-radius:20px;
    padding:3px 10px; font-size:.72rem; color:white;
  }
  /* Chat */
  .chat-bubble-user {
    background:#FF6B35; color:white; border-radius:16px 16px 4px 16px;
    padding:10px 16px; margin:6px 0 6px 40px; font-size:.88rem;
  }
  .chat-bubble-bot {
    background:rgba(255,255,255,.07); color:white;
    border-radius:16px 16px 16px 4px;
    padding:10px 16px; margin:6px 40px 6px 0; font-size:.88rem;
    border-left:3px solid #FF6B35;
  }
  
  /* ── CNBC Ticker ── */
  .cnbc-ticker-wrapper {
    width: 100%; overflow: hidden; position: relative;
    border-bottom: 2px solid #000; margin-bottom: 24px;
    font-family: 'Syne', sans-serif;
  }
  .cnbc-ticker-strip {
    display: flex; white-space: nowrap; align-items: center; padding: 6px 0;
    width: max-content;
  }
  .strip-white {
    background: #ffffff; color: #000; border-top: 3px solid #0033a0;
    animation: scroll-left-fast 90s linear infinite;
  }
  .strip-blue {
    background: #002244; color: #fff; border-top: 1px solid #444;
    animation: scroll-left-slow 100s linear infinite;
  }
  .ticker-item {
    display: inline-flex; align-items: center; padding: 0 20px;
    border-right: 1px solid rgba(128,128,128,0.3); font-size: 1.05rem; font-weight: 700;
  }
  .ti-name { margin-right: 8px; letter-spacing: 0.5px; }
  .strip-white .ti-name { color: #0033a0; }
  .strip-blue .ti-name { color: #fff; }
  .ti-vol { font-size: 0.8rem; font-weight: 600; color: #777; margin-right: 12px; }
  .ti-price { margin-right: 8px; font-weight: 800; }
  .ti-up { color: #008800; }
  .strip-blue .ti-up { color: #00ff88; }
  .ti-down { color: #cc0000; }
  .strip-blue .ti-down { color: #ff4444; }
  .ti-arrow { font-size: 0.9rem; margin-right: 4px; }
  
  @keyframes scroll-left-fast { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
  @keyframes scroll-left-slow { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
  
  /* ── Hot News Tijori Style ── */
  .hot-news-container {
    background: #1a1a1a; border-radius: 12px; padding: 16px 20px;
    margin-bottom: 24px; border-left: 5px solid #FF6B35;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    display: flex; flex-direction: column; gap: 12px;
  }
  .hot-news-list {
    display: flex; flex-direction: column; gap: 12px;
    max-height: 380px; overflow-y: auto; padding-right: 8px;
  }
  .hot-news-list::-webkit-scrollbar { width: 6px; }
  .hot-news-list::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 4px; }
  .hot-news-list::-webkit-scrollbar-thumb { background: rgba(255,107,53,0.5); border-radius: 4px; }
  .hn-header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;
  }
  .hn-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800; color: white;
    display: flex; align-items: center; gap: 8px;
  }
  .hn-badge {
    background: rgba(255,107,53,0.2); color: #FF6B35; font-size: 0.7rem;
    padding: 3px 8px; border-radius: 20px; font-weight: 700; text-transform: uppercase;
  }
  .hn-item {
    background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px 16px;
    transition: background 0.2s;
  }
  .hn-item:hover {
    background: rgba(255,255,255,0.06);
  }
  .hn-time {
    font-size: 0.7rem; color: #888; font-family: monospace;
    margin-bottom: 6px; display: block;
  }
  .hn-gist {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; color: #FFD700;
    margin-bottom: 4px;
  }
  .hn-summary {
    font-size: 0.85rem; color: rgba(255,255,255,0.85); line-height: 1.5; margin-bottom: 8px;
  }
  .hn-link {
    font-size: 0.75rem; color: #00ff88; text-decoration: none; font-weight: 600;
  }
  .hn-link:hover { text-decoration: underline; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────── Hero ──────────────────────────────────────────

st.markdown(
    """
<div class="hero-header">
  <h1>🇮🇳 Indian Market Research Agent</h1>
  <p>AI market intelligence • Candlestick charts • Bulkowski AI prediction engine • Groq Llama 3.3 70B</p>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
#  LIVE TICKER STRIP
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_cnbc_ticker_data():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "LT.NS", 
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
    except Exception:
        return []

def render_cnbc_ticker():
    data = fetch_cnbc_ticker_data()
    if not data or len(data) < 4:
        return ""
        
    half = len(data) // 2
    top_data = data[:half] * 8 # Repeat for smooth infinite scroll
    bot_data = data[half:] * 8
    
    def _build_strip(items, cls):
        html = f'<div class="cnbc-ticker-strip {cls}">'
        for it in items:
            is_up = it['chg'] >= 0
            c_cls = "ti-up" if is_up else "ti-down"
            arr = "▲" if is_up else "▼"
            html += f"""<div class="ticker-item">
<span class="ti-name">{it['name']}</span>
<span class="ti-vol">{it['vol']}</span>
<span class="ti-price">{it['price']:.2f}</span>
<span class="{c_cls}"><span class="ti-arrow">{arr}</span>{abs(it['chg']):.2f}</span>
</div>"""
        html += "</div>"
        return html

    top_strip = _build_strip(top_data, "strip-white")
    bot_strip = _build_strip(bot_data, "strip-blue")
    
    return f"""<div class="cnbc-ticker-wrapper">
{top_strip}
{bot_strip}
</div>"""

st.markdown(render_cnbc_ticker(), unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_news():
    try:
        return news_engine.fetch_and_summarize_news()
    except Exception:
        return []

def render_hot_news():
    # Automatically refresh the Streamlit app every 5 minutes (300,000 ms)
    st_autorefresh(interval=300000, key="news_autorefresh")
    
    news_items = get_cached_news()
    if not news_items:
        return ""
    
    items_html = ""
    for item in news_items:
        # Some RSS feeds send very long date strings, slice to keep it clean
        time_str = item.get('time', 'Recently')
        if len(time_str) > 22:
            time_str = time_str[:22] + ".."
            
        items_html += f"""<div class="hn-item">
<span class="hn-time">🕒 {time_str}</span>
<div class="hn-gist">⚡ {item.get('gist', 'Breaking News')}</div>
<div class="hn-summary">{item.get('summary', item.get('title', ''))}</div>
<a href="{item.get('link', '#')}" target="_blank" class="hn-link">🔗 Read Details</a>
</div>"""
        
    return f"""<div class="hot-news-container">
<div class="hn-header">
<div class="hn-title">📰 Live Market Updates <span class="hn-badge">AI Summarized</span></div>
<div style="font-size:0.75rem; color:#aaa;">Updates from ET Markets</div>
</div>
<div class="hot-news-list">
{items_html}
</div>
</div>"""

st.markdown(render_hot_news(), unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🔍  Market Research",
        "📈  Portfolio Tracker",
        "📊  Ticker Board",
        "🌐  NSE Market Watch",
        "🎯  Smart Recommendations",
        "🤖  RL Simulator",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PORTFOLIO_FILE = "portfolio.json"

QUICK_ADD_STOCKS = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("TCS", "TCS.NS"),
    ("Infosys", "INFY.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("Tata Motors", "TATAMOTORS.NS"),
    ("Zomato", "ZOMATO.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("Wipro", "WIPRO.NS"),
]


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

def get_session():
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session


@st.cache_data(ttl=60, show_spinner=False)
def fetch_portfolio_snapshots_cached(portfolio_entries_json):
    portfolio_entries = json.loads(portfolio_entries_json)
    snapshots = {}
    tickers = list(set([entry.get("ticker", "UNKNOWN") for entry in portfolio_entries if entry.get("ticker")]))
    if not tickers:
        return {}

    raw_data = None
    err_msg = "Unknown error"
    
    # Download in one single batch HTTP request (drastically reduces 429 risk)
    try:
        raw_data = yf.download(
            tickers,
            period="5y",
            group_by="ticker",
            progress=False,
            threads=True
        )
    except Exception as e:
        err_msg = str(e)

    for entry in portfolio_entries:
        eid = entry.get("id", entry.get("ticker", "UNKNOWN"))
        ticker = entry.get("ticker", "UNKNOWN")
        company = entry.get("company", "Unknown")
        try:
            if raw_data is None or raw_data.empty:
                raise ValueError(f"Failed to fetch data from Yahoo Finance: {err_msg}")

            # Extract this ticker's df
            if len(tickers) == 1:
                df_t = raw_data
            else:
                if hasattr(raw_data.columns, 'levels') and ticker in raw_data.columns.levels[0]:
                    df_t = raw_data[ticker]
                elif hasattr(raw_data.columns, 'levels') and ticker in raw_data.columns.get_level_values(0):
                    df_t = raw_data[ticker]
                else:
                    raise ValueError(f"No data found for {ticker}")

            if df_t is None or df_t.empty:
                raise ValueError(f"No data found for {ticker}")

            df_t = df_t.dropna(subset=["Close"])
            if len(df_t) < 2:
                raise ValueError(f"Insufficient data for {ticker}")

            hist_close = float(df_t["Close"].iloc[-1])
            prev_close = float(df_t["Close"].iloc[-2])

            if np.isnan(hist_close) or np.isnan(prev_close) or hist_close <= 0:
                raise ValueError(f"Invalid price data for {ticker}")

            # Try to get live price via fast_info fallback or yf.Ticker
            cur = hist_close
            prev = prev_close
            try:
                stock_obj = yf.Ticker(ticker)
                fi = stock_obj.fast_info
                live_price = float(fi.last_price)
                prev_close_ = float(fi.previous_close) if fi.previous_close else prev_close
                if not np.isnan(live_price) and live_price > 0:
                    cur = live_price
                    prev = prev_close_ if (prev_close_ and not np.isnan(prev_close_)) else prev_close
            except Exception:
                pass

            chg = (cur - prev) / prev * 100 if prev else 0

            # Technical indicators
            ma20 = float(df_t["Close"].rolling(20).mean().iloc[-1]) if len(df_t) >= 20 else None
            ma50 = float(df_t["Close"].rolling(50).mean().iloc[-1]) if len(df_t) >= 50 else None
            if ma20 and np.isnan(ma20): ma20 = None
            if ma50 and np.isnan(ma50): ma50 = None

            rsi = None
            if len(df_t) >= 15:
                d = df_t["Close"].diff()
                g = d.clip(lower=0).rolling(14).mean()
                l = (-d.clip(upper=0)).rolling(14).mean()
                r = g / l.replace(0, float("nan"))
                v = (100 - 100 / (1 + r)).iloc[-1]
                rsi = float(v) if not np.isnan(v) else None

            bull, bear = 0, 0
            if ma20:
                bull += 1 if cur > ma20 else 0
                bear += 1 if cur < ma20 else 0
            if ma50:
                bull += 1 if cur > ma50 else 0
                bear += 1 if cur < ma50 else 0
            if rsi:
                if rsi < 35: bull += 1
                if rsi > 65: bear += 1
            signal = "BULLISH" if bull > bear + 1 else ("BEARISH" if bear > bull + 1 else "NEUTRAL")

            vol_series = df_t["Volume"].replace(0, np.nan).dropna()
            avg_vol = int(vol_series.rolling(min(20, len(vol_series))).mean().iloc[-1]) if len(vol_series) >= 2 else 0
            today_vol = int(df_t["Volume"].iloc[-1]) if not np.isnan(df_t["Volume"].iloc[-1]) else 0

            snapshots[eid] = {
                "ticker": ticker,
                "company": company,
                "current_price": round(cur, 2),
                "prev_close": round(prev, 2),
                "today_change": round(chg, 2),
                "week52_high": round(float(df_t["High"].tail(252).max()), 2),
                "week52_low": round(float(df_t["Low"].tail(252).min()), 2),
                "ma20": round(ma20, 2) if ma20 else None,
                "ma50": round(ma50, 2) if ma50 else None,
                "rsi": round(rsi, 2) if rsi else None,
                "signal": signal,
                "avg_volume": avg_vol,
                "today_volume": today_vol,
                "return_1m": _pret(df_t, 21),
                "df": df_t,
            }
        except Exception as e:
            snapshots[eid] = {"error": str(e), "ticker": ticker, "company": company}

    return snapshots


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_stock_snapshot_cached(company, ticker):
    time.sleep(0.5)  # Add small delay to prevent batch rate-limiting
    stock = yf.Ticker(ticker)

    # ── Historical data for indicators (5 Years) ──────────────────
    df = stock.history(period="5y")
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")

    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        raise ValueError(f"Insufficient data for {ticker}")

    hist_close = float(df["Close"].iloc[-1])  # last known close (may be yesterday)
    prev_close = float(df["Close"].iloc[-2])

    if np.isnan(hist_close) or np.isnan(prev_close) or hist_close <= 0:
        raise ValueError(f"Invalid price data for {ticker}")

    # ── Real-time price via fast_info ─────────────────────────────
    # fast_info.last_price reflects intraday price during market hours
    try:
        fi = stock.fast_info
        live_price = float(fi.last_price)
        prev_close_ = float(fi.previous_close) if fi.previous_close else prev_close

        # Sanity check — if fast_info returns garbage, fall back
        if np.isnan(live_price) or live_price <= 0:
            raise ValueError("bad fast_info price")

        cur = live_price
        prev = (
            prev_close_
            if (prev_close_ and not np.isnan(prev_close_))
            else prev_close
        )
    except Exception:
        # Fallback to last historical close
        cur = hist_close
        prev = prev_close

    chg = (cur - prev) / prev * 100 if prev else 0

    # ── Technical indicators (from historical df) ─────────────────
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
    if ma20 and np.isnan(ma20):
        ma20 = None
    if ma50 and np.isnan(ma50):
        ma50 = None

    rsi = None
    if len(df) >= 15:
        d = df["Close"].diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        r = g / l.replace(0, float("nan"))
        v = (100 - 100 / (1 + r)).iloc[-1]
        rsi = float(v) if not np.isnan(v) else None

    bull, bear = 0, 0
    if ma20:
        bull += 1 if cur > ma20 else 0
        bear += 1 if cur < ma20 else 0
    if ma50:
        bull += 1 if cur > ma50 else 0
        bear += 1 if cur < ma50 else 0
    if rsi:
        if rsi < 35:
            bull += 1
        if rsi > 65:
            bear += 1
    signal = (
        "BULLISH"
        if bull > bear + 1
        else ("BEARISH" if bear > bull + 1 else "NEUTRAL")
    )

    vol_series = df["Volume"].replace(0, np.nan).dropna()
    avg_vol = (
        int(vol_series.rolling(min(20, len(vol_series))).mean().iloc[-1])
        if len(vol_series) >= 2
        else 0
    )
    today_vol = (
        int(df["Volume"].iloc[-1]) if not np.isnan(df["Volume"].iloc[-1]) else 0
    )

    return {
        "ticker": ticker,
        "company": company,
        "current_price": round(cur, 2),
        "prev_close": round(prev, 2),
        "today_change": round(chg, 2),
        "week52_high": round(float(df["High"].tail(252).max()), 2),
        "week52_low": round(float(df["Low"].tail(252).min()), 2),
        "ma20": round(ma20, 2) if ma20 else None,
        "ma50": round(ma50, 2) if ma50 else None,
        "rsi": round(rsi, 2) if rsi else None,
        "signal": signal,
        "avg_volume": avg_vol,
        "today_volume": today_vol,
        "return_1m": _pret(df, 21),
        "df": df,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_hourly_data_cached(ticker):
    try:
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo", interval="1h")
        if not df.empty:
            df = df.dropna(subset=["Close"])
            return df
    except Exception:
        pass
    return None


def fetch_stock_snapshot(company, ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return _fetch_stock_snapshot_cached(company, ticker)
        except Exception as e:
            if attempt < max_retries - 1:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "too many requests" in err_str:
                    time.sleep(2 ** attempt + 1.5)
                else:
                    time.sleep(1)
            else:
                return {"error": str(e), "ticker": ticker, "company": company}
    return {"error": "Max retries exceeded", "ticker": ticker, "company": company}


def _pret(df, days):
    if len(df) < 2:
        return None
    lb = min(days, len(df) - 1)
    p = float(df["Close"].iloc[-lb - 1])
    n = float(df["Close"].iloc[-1])
    if np.isnan(p) or np.isnan(n) or p == 0:
        return None
    return round((n - p) / p * 100, 2)


def _signal_badge(s):
    m = {"BULLISH": ("badge-bull", "▲ BULLISH"), "BEARISH": ("badge-bear", "▼ BEARISH")}
    cls, lbl = m.get(s, ("badge-neutral", "◆ NEUTRAL"))
    return f'<span class="{cls}">{lbl}</span>'


def _ppill(p):
    t = p.get("type", "neutral")
    cls = {"bullish": "ppill-bull", "bearish": "ppill-bear"}.get(t, "ppill-neutral")
    return f'<span class="{cls}">{p["name"]}</span>'


def build_tradingview_lightweight_chart(
    df: pd.DataFrame, ticker: str, prediction: dict = None, days: int = 90
) -> str:
    """
    Build beautiful TradingView Lightweight Charts HTML code with MA overlays,
    volume histograms, and custom AI price target, support, and resistance overlays.
    """
    import json
    
    df_chart = df.tail(days).copy()
    df_chart.index = pd.to_datetime(df_chart.index)
    
    # Calculate MAs
    ma20 = df_chart["Close"].rolling(20).mean()
    ma50 = df_chart["Close"].rolling(50).mean()
    
    # Prepare candles, volume, and MA data lists
    candles_data = []
    volume_data = []
    ma20_data = []
    ma50_data = []
    
    for idx, row in df_chart.iterrows():
        t_val = int(idx.timestamp()) # Use UNIX timestamp to prevent duplicate date crashes in hourly swing data
        c_val = float(row["Close"])
        o_val = float(row["Open"])
        h_val = float(row["High"])
        l_val = float(row["Low"])
        v_val = float(row["Volume"]) if "Volume" in row else 0.0
        
        candles_data.append({
            "time": t_val,
            "open": o_val,
            "high": h_val,
            "low": l_val,
            "close": c_val
        })
        
        # Color volume bars green if close >= open, else red
        v_color = "rgba(0, 208, 132, 0.35)" if c_val >= o_val else "rgba(255, 68, 68, 0.35)"
        volume_data.append({
            "time": t_val,
            "value": v_val,
            "color": v_color
        })
        
        m20 = ma20.loc[idx]
        if pd.notna(m20):
            ma20_data.append({"time": t_val, "value": float(m20)})
            
        m50 = ma50.loc[idx]
        if pd.notna(m50):
            ma50_data.append({"time": t_val, "value": float(m50)})
            
    # Serialize to JSON strings for easy JS parsing
    candles_json = json.dumps(candles_data)
    volume_json = json.dumps(volume_data)
    ma20_json = json.dumps(ma20_data)
    ma50_json = json.dumps(ma50_data)
    
    # Extract prediction lines
    pred_data = {}
    if prediction:
        pred_data = {
            "target": prediction.get("price_target"),
            "direction": prediction.get("direction", "NEUTRAL"),
            "support": prediction.get("support_level"),
            "resistance": prediction.get("resistance_level")
        }
    pred_json = json.dumps(pred_data)
    
    # Complete HTML block with embedded CSS and Lightweight Charts CDN
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background-color: #0e1117;
                overflow: hidden;
            }}
            #chart-container {{
                width: 100%;
                height: 380px; /* Fixed pixel height to prevent layout collapse in Streamlit */
                box-sizing: border-box;
            }}
            /* Tooltip style */
            .chart-tooltip {{
                position: absolute;
                display: none;
                padding: 8px 12px;
                box-sizing: border-box;
                font-size: 11px;
                text-align: left;
                z-index: 1000;
                top: 10px;
                left: 10px;
                pointer-events: none;
                border: 1px solid rgba(255, 107, 53, 0.4);
                border-radius: 6px;
                background: rgba(20, 20, 20, 0.9);
                color: #fff;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            }}
            .tooltip-title {{
                font-weight: bold;
                color: #FF6B35;
                margin-bottom: 4px;
                font-size: 12px;
            }}
            .tooltip-row {{
                display: flex;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 2px;
            }}
        </style>
        <!-- Stable and fast jsDelivr CDN to load TradingView Lightweight Charts -->
        <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    </head>
    <body>
        <div id="chart-container"></div>
        <div id="tooltip" class="chart-tooltip"></div>
        
        <script>
            const container = document.getElementById('chart-container');
            const tooltip = document.getElementById('tooltip');
            
            const chartOptions = {{
                layout: {{
                    textColor: '#c5cbce',
                    background: {{ type: 'solid', color: '#0e1117' }},
                    fontSize: 11,
                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                    horzLines: {{ color: 'rgba(255, 255, 255, 0.04)' }},
                }},
                rightPriceScale: {{
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    visible: true,
                }},
                timeScale: {{
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    timeVisible: true,
                    secondsVisible: false,
                }},
                crosshair: {{
                    mode: 1, // Magnet mode
                    vertLine: {{
                        color: 'rgba(255, 255, 255, 0.25)',
                        width: 1,
                        style: 3 // Dotted
                    }},
                    horzLine: {{
                        color: 'rgba(255, 255, 255, 0.25)',
                        width: 1,
                        style: 3 // Dotted
                    }}
                }},
            }};
            
            const chart = LightweightCharts.createChart(container, chartOptions);
            
            // 1. Candlestick series
            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#00d084',
                downColor: '#ff4444',
                borderVisible: false,
                wickUpColor: '#00d084',
                wickDownColor: '#ff4444',
            }});
            
            candlestickSeries.setData({candles_json});
            
            // 2. Volume series (Assigned to custom scale ID 'volume' to prevent assertion errors)
            const volumeSeries = chart.addHistogramSeries({{
                priceFormat: {{
                    type: 'volume',
                }},
                priceScaleId: 'volume', // Using custom non-empty scale ID
            }});
            
            // Set position and margins for the custom volume scale
            chart.priceScale('volume').applyOptions({{
                scaleMargins: {{
                    top: 0.8, // bottom 20%
                    bottom: 0,
                }},
                visible: false, // hide the axis numbers
            }});
            
            volumeSeries.setData({volume_json});
            
            // 3. MA20 (Orange line)
            const ma20Series = chart.addLineSeries({{
                color: '#FF9A3C',
                lineWidth: 1.5,
                title: 'MA20',
                crosshairMarkerVisible: false,
                lastValueVisible: false,
            }});
            ma20Series.setData({ma20_json});
            
            // 4. MA50 (Purple line)
            const ma50Series = chart.addLineSeries({{
                color: '#9b59b6',
                lineWidth: 1.5,
                lineStyle: 1, // Dashed
                title: 'MA50',
                crosshairMarkerVisible: false,
                lastValueVisible: false,
            }});
            ma50Series.setData({ma50_json});
            
            // 5. Prediction lines (Added strict type checks to prevent NaN errors)
            const pred = {pred_json};
            if (pred && typeof pred.target === 'number' && !isNaN(pred.target)) {{
                let targetColor = '#ffd700'; // neutral
                let targetRGBA = 'rgba(255, 215, 0, 0.4)';
                let lineRGBA = 'rgba(255, 215, 0, 0.75)';
                if (pred.direction === 'BULLISH') {{
                    targetColor = '#00ff88';
                    targetRGBA = 'rgba(0, 255, 136, 0.4)';
                    lineRGBA = 'rgba(0, 255, 136, 0.75)';
                }}
                if (pred.direction === 'BEARISH') {{
                    targetColor = '#ff4444';
                    targetRGBA = 'rgba(255, 68, 68, 0.4)';
                    lineRGBA = 'rgba(255, 68, 68, 0.75)';
                }}
                
                // Add soft horizontal target guideline (subtle, thin dashed line)
                candlestickSeries.createPriceLine({{
                    price: pred.target,
                    color: targetRGBA,
                    lineWidth: 1,
                    lineStyle: 2, // Dashed
                    axisLabelVisible: true,
                    title: '🎯 Target (₹' + pred.target.toFixed(2) + ')',
                }});
                
                // Draw diagonal forecast projection line starting from last candle close to the future target
                const lastCandle = candleData[candleData.length - 1];
                if (lastCandle) {{
                    const lastTime = lastCandle.time;
                    
                    // Determine timeframe distance in days
                    let daysDelta = 5;
                    const tf = pred.timeframe.toLowerCase();
                    if (tf.includes("day") || tf.includes("swing")) daysDelta = 3;
                    else if (tf.includes("week") || tf.includes("short")) daysDelta = 10;
                    else if (tf.includes("month") || tf.includes("medium")) daysDelta = 30;
                    else if (tf.includes("long")) daysDelta = 120;
                    
                    const futureTime = lastTime + (daysDelta * 24 * 60 * 60);
                    
                    const forecastSeries = chart.addLineSeries({{
                        color: lineRGBA,
                        lineWidth: 1.5,
                        lineStyle: 3, // Dotted
                        title: 'AI Forecast',
                        crosshairMarkerVisible: false,
                        lastValueVisible: false,
                    }});
                    
                    forecastSeries.setData([
                        {{ time: lastTime, value: lastCandle.close }},
                        {{ time: futureTime, value: pred.target }}
                    ]);
                }}
                
                // Add support line (very soft dotted line)
                if (typeof pred.support === 'number' && !isNaN(pred.support)) {{
                    candlestickSeries.createPriceLine({{
                        price: pred.support,
                        color: 'rgba(0, 208, 132, 0.25)',
                        lineWidth: 1,
                        lineStyle: 3, // Dotted
                        axisLabelVisible: true,
                        title: 'Support (₹' + pred.support.toFixed(2) + ')',
                    }});
                }}
                
                // Add resistance line (very soft dotted line)
                if (typeof pred.resistance === 'number' && !isNaN(pred.resistance)) {{
                    candlestickSeries.createPriceLine({{
                        price: pred.resistance,
                        color: 'rgba(255, 68, 68, 0.25)',
                        lineWidth: 1,
                        lineStyle: 3, // Dotted
                        axisLabelVisible: true,
                        title: 'Resistance (₹' + pred.resistance.toFixed(2) + ')',
                    }});
                }}
            }}
            
            // Tooltip handler
            chart.subscribeCrosshairMove(param => {{
                if (
                    param.point === undefined ||
                    !param.time ||
                    param.point.x < 0 ||
                    param.point.y < 0
                ) {{
                    tooltip.style.display = 'none';
                    return;
                }}
                
                const data = param.seriesData.get(candlestickSeries);
                if (!data) {{
                    tooltip.style.display = 'none';
                    return;
                }}
                
                const o = data.open !== undefined ? data.open : 0;
                const h = data.high !== undefined ? data.high : 0;
                const l = data.low !== undefined ? data.low : 0;
                const c = data.close !== undefined ? data.close : 0;
                
                // Format date and time nicely
                let dateStr = param.time;
                if (typeof dateStr === 'number') {{
                    // UNIX timestamp in seconds -> JS millisecond Date
                    const d = new Date(dateStr * 1000);
                    // Check if time is relevant (non-zero) to format hourly vs daily
                    const hasTime = d.getHours() !== 0 || d.getMinutes() !== 0;
                    const options = {{ 
                        day: 'numeric', 
                        month: 'short', 
                        year: 'numeric'
                    }};
                    if (hasTime) {{
                        options.hour = '2-digit';
                        options.minute = '2-digit';
                        options.hour12 = true;
                    }}
                    dateStr = d.toLocaleDateString('en-IN', options);
                }}
                
                tooltip.style.display = 'block';
                tooltip.innerHTML = `
                    <div class="tooltip-title">{ticker}</div>
                    <div class="tooltip-row"><span>Date:</span><span><b>${{dateStr}}</b></span></div>
                    <div class="tooltip-row"><span>Open:</span><span style="font-family: monospace;">₹${{o.toFixed(2)}}</span></div>
                    <div class="tooltip-row"><span>High:</span><span style="font-family: monospace; color: #00d084;">₹${{h.toFixed(2)}}</span></div>
                    <div class="tooltip-row"><span>Low:</span><span style="font-family: monospace; color: #ff4444;">₹${{l.toFixed(2)}}</span></div>
                    <div class="tooltip-row"><span>Close:</span><span style="font-family: monospace; font-weight: bold; color: #FFD700;">₹${{c.toFixed(2)}}</span></div>
                `;
            }});
            
            // Resize handler
            const resizeObserver = new ResizeObserver(entries => {{
                if (entries.length === 0 || !entries[0].contentRect) return;
                const {{ width, height }} = entries[0].contentRect;
                chart.resize(width, height);
            }});
            resizeObserver.observe(container);
        </script>
    </body>
    </html>
    """
    return html_code


# ─────────────────────────── Smart Action Advisor ──────────────────────────


def compute_smart_action(entry: dict, snap: dict, pred_result: dict = None) -> dict:
    """
    Smart Action Advisor — analyzes portfolio holdings and recommends:
    BUY MORE, HOLD, or EXIT with reasoning, targets, and opportunity analysis.

    Uses: RSI, MA20/MA50 positioning, 52W range, trend, P&L%, prediction engine.
    """
    price = float(snap["current_price"])
    avg_buy = float(entry["avg_price"])
    qty = int(entry["quantity"])
    rsi = float(snap.get("rsi", 50.0) or 50.0)
    ma20 = float(snap.get("ma20", 0) or 0)
    ma50 = float(snap.get("ma50", 0) or 0)
    w52_high = float(snap.get("week52_high", price))
    w52_low = float(snap.get("week52_low", price))
    return_1m = float(snap.get("return_1m", 0) or 0)
    signal = snap.get("signal", "NEUTRAL")

    # ── P&L analysis ──
    pnl_pct = (price - avg_buy) / avg_buy * 100.0 if avg_buy > 0 else 0.0
    inv_value = qty * avg_buy
    cur_value = qty * price
    pnl_rupees = cur_value - inv_value

    # ── 52W position (0% = at 52W low, 100% = at 52W high) ──
    w52_range = w52_high - w52_low if w52_high > w52_low else 1.0
    w52_position = (price - w52_low) / w52_range * 100.0

    # ── Scoring system for action recommendation ──
    # Positive score = BUY MORE, Negative score = EXIT
    action_score = 0.0
    reasons_buy = []
    reasons_exit = []
    reasons_hold = []

    # RSI analysis
    if rsi < 30:
        action_score += 30
        reasons_buy.append(f"RSI at {rsi:.1f} — heavily oversold, strong bounce potential")
    elif rsi < 40:
        action_score += 15
        reasons_buy.append(f"RSI at {rsi:.1f} — oversold territory, buying opportunity")
    elif rsi > 75:
        action_score -= 30
        reasons_exit.append(f"RSI at {rsi:.1f} — heavily overbought, correction likely")
    elif rsi > 65:
        action_score -= 10
        reasons_exit.append(f"RSI at {rsi:.1f} — nearing overbought, momentum fading")
    else:
        reasons_hold.append(f"RSI at {rsi:.1f} — neutral zone")

    # MA positioning
    if ma20 > 0 and ma50 > 0:
        if price > ma20 and price > ma50:
            action_score += 10
            reasons_buy.append("Price above both MA20 & MA50 — uptrend intact")
        elif price < ma20 and price < ma50:
            action_score -= 15
            reasons_exit.append("Price below both MA20 & MA50 — downtrend pressure")
        elif price > ma50 and price < ma20:
            reasons_hold.append("Price between MA50 (support) and MA20 — consolidating")
        if ma20 > ma50:
            action_score += 5
            reasons_buy.append("MA20 > MA50 — golden cross / bullish structure")
        elif ma20 < ma50 * 0.98:
            action_score -= 5
            reasons_exit.append("MA20 < MA50 — death cross / bearish structure")

    # 52W range positioning
    if w52_position < 15:
        action_score += 20
        reasons_buy.append(f"Near 52W low ({w52_position:.0f}% from bottom) — deep value zone")
    elif w52_position < 30:
        action_score += 10
        reasons_buy.append(f"In lower 30% of 52W range — value territory")
    elif w52_position > 90:
        action_score -= 20
        reasons_exit.append(f"Near 52W high ({w52_position:.0f}% of range) — limited upside")
    elif w52_position > 75:
        action_score -= 10
        reasons_exit.append(f"In upper 25% of 52W range — approaching resistance ceiling")

    # 1-month return momentum
    if return_1m < -8:
        action_score += 10
        reasons_buy.append(f"1M return {return_1m:+.1f}% — recent dip = buying opportunity")
    elif return_1m > 15:
        action_score -= 10
        reasons_exit.append(f"1M return {return_1m:+.1f}% — sharp run-up, profit booking zone")

    # P&L-based profit booking signals
    if pnl_pct > 50:
        action_score -= 10
        reasons_exit.append(f"Already +{pnl_pct:.1f}% profit — consider partial profit booking")
    elif pnl_pct > 25:
        action_score -= 5
        reasons_exit.append(f"+{pnl_pct:.1f}% profit — set trailing stop loss to protect gains")
    elif pnl_pct < -15:
        reasons_hold.append(f"In {pnl_pct:.1f}% loss — averaging down may be risky without trend reversal")

    # Prediction engine data (if available)
    pred_direction = None
    pred_target = None
    pred_confidence = None
    if pred_result and "error" not in pred_result:
        pred = pred_result.get("prediction", {})
        quant = pred_result.get("quant_prediction", {})
        pred_direction = pred.get("direction", quant.get("direction"))
        pred_target = pred.get("price_target", quant.get("price_target"))
        pred_confidence = pred.get("confidence", quant.get("confidence", 50))

        if pred_direction == "BULLISH" and pred_confidence and pred_confidence > 70:
            action_score += 15
            reasons_buy.append(f"AI prediction: BULLISH ({pred_confidence:.0f}% confidence)")
        elif pred_direction == "BEARISH" and pred_confidence and pred_confidence > 70:
            action_score -= 15
            reasons_exit.append(f"AI prediction: BEARISH ({pred_confidence:.0f}% confidence)")

    # ── Determine final action ──
    if action_score >= 20:
        action = "BUY MORE"
        action_icon = "🟢"
        action_color = "#00c853"
        action_bg = "rgba(0,200,83,0.08)"
        action_border = "rgba(0,200,83,0.4)"
    elif action_score <= -20:
        action = "EXIT / BOOK PROFIT"
        action_icon = "🔴"
        action_color = "#ff1744"
        action_bg = "rgba(255,23,68,0.08)"
        action_border = "rgba(255,23,68,0.4)"
    else:
        action = "HOLD"
        action_icon = "🟡"
        action_color = "#ffd600"
        action_bg = "rgba(255,214,0,0.08)"
        action_border = "rgba(255,214,0,0.4)"

    # ── Compute opportunity analysis ──
    # If you bought more at avg_buy, how much would you have gained?
    opportunity_pct = pnl_pct  # same as current P&L%
    # What if you invested ₹10,000 more at current levels?
    hypothetical_invest = 10000.0
    hypothetical_qty = int(hypothetical_invest / price) if price > 0 else 0

    # ── Compute target exit price ──
    if pred_target and pred_target > 0:
        exit_target = float(pred_target)
    elif action == "EXIT / BOOK PROFIT":
        # Use resistance or a small buffer above current
        exit_target = price * 1.02  # exit near current if bearish
    else:
        # Use resistance as target
        exit_target = w52_high * 0.95 if w52_high > price else price * 1.10

    # Potential gain if bought now and exited at target
    if_bought_now_gain = (exit_target - price) / price * 100.0 if price > 0 else 0
    if_bought_now_rupees = hypothetical_qty * (exit_target - price) if hypothetical_qty > 0 else 0

    return {
        "action": action,
        "action_icon": action_icon,
        "action_color": action_color,
        "action_bg": action_bg,
        "action_border": action_border,
        "action_score": action_score,
        "reasons_buy": reasons_buy,
        "reasons_exit": reasons_exit,
        "reasons_hold": reasons_hold,
        "pnl_pct": pnl_pct,
        "pnl_rupees": pnl_rupees,
        "opportunity_pct": opportunity_pct,
        "w52_position": w52_position,
        "exit_target": round(exit_target, 2),
        "if_bought_now_gain": round(if_bought_now_gain, 2),
        "if_bought_now_rupees": round(if_bought_now_rupees, 2),
        "hypothetical_qty": hypothetical_qty,
        "hypothetical_invest": hypothetical_invest,
        "pred_direction": pred_direction,
        "pred_confidence": pred_confidence,
    }


def render_smart_action(entry: dict, snap: dict, smart: dict) -> str:
    """Render the Smart Action Advisor card as HTML."""
    price = snap["current_price"]
    avg_buy = entry["avg_price"]
    qty = entry["quantity"]

    action = smart["action"]
    icon = smart["action_icon"]
    color = smart["action_color"]
    bg = smart["action_bg"]
    border = smart["action_border"]

    # Build reason bullets
    all_reasons = []
    for r in smart["reasons_buy"]:
        all_reasons.append(f"<span style='color:#00ff88;'>▲</span> {r}")
    for r in smart["reasons_exit"]:
        all_reasons.append(f"<span style='color:#ff4444;'>▼</span> {r}")
    for r in smart["reasons_hold"]:
        all_reasons.append(f"<span style='color:#ffd600;'>●</span> {r}")
    reasons_html = "<br>".join(all_reasons[:5])  # Show top 5 reasons

    # Opportunity section
    pnl_pct = smart["pnl_pct"]
    if pnl_pct > 0:
        opp_text = (
            f"You're up <b style='color:#00ff88;'>{pnl_pct:+.2f}%</b> from your avg buy of ₹{avg_buy:,.2f}. "
        )
    else:
        opp_text = (
            f"Currently at <b style='color:#ff4444;'>{pnl_pct:+.2f}%</b> from your avg buy of ₹{avg_buy:,.2f}. "
        )

    hyp_qty = smart["hypothetical_qty"]
    hyp_gain = smart["if_bought_now_gain"]
    hyp_rupees = smart["if_bought_now_rupees"]
    exit_target = smart["exit_target"]

    if hyp_qty > 0 and hyp_gain > 0:
        if_buy_html = (
            f"If you invest <b>₹{smart['hypothetical_invest']:,.0f}</b> more now "
            f"(≈{hyp_qty} shares at ₹{price:,.2f}), and exit at target <b>₹{exit_target:,.2f}</b>, "
            f"potential gain: <b style='color:#00ff88;'>₹{hyp_rupees:,.0f} (+{hyp_gain:.2f}%)</b>"
        )
    elif hyp_gain <= 0:
        if_buy_html = (
            f"Target price ₹{exit_target:,.2f} is below current price — "
            f"<b style='color:#ff4444;'>not a good entry point right now</b>"
        )
    else:
        if_buy_html = ""

    # 52W position bar
    w52_pos = smart["w52_position"]
    bar_color = "#00ff88" if w52_pos < 40 else ("#ffd600" if w52_pos < 70 else "#ff4444")

    return f"""<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:16px 20px;margin:12px 0;border-left:5px solid {color};">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:0.7rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px;color:white;">🧠 Smart Action Advisor</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{color};margin-top:4px;">{icon} {action}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.75rem;color:#aaa;">Target Exit Price</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:white;">🎯 ₹{exit_target:,.2f}</div>
    <div style="font-size:0.72rem;color:#888;">Score: {smart['action_score']:+.0f}</div>
  </div>
</div>

<div style="margin-top:12px;font-size:0.82rem;color:rgba(255,255,255,0.9);line-height:1.7;">
  {reasons_html}
</div>

<div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
  <div style="flex:1;min-width:200px;background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.68rem;color:#aaa;text-transform:uppercase;margin-bottom:4px;">📊 52W Position</div>
    <div style="width:100%;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
      <div style="width:{w52_pos:.0f}%;height:100%;background:{bar_color};border-radius:4px;"></div>
    </div>
    <div style="font-size:0.72rem;color:#ccc;margin-top:4px;">{w52_pos:.1f}% from 52W low → {'Near bottom 🟢' if w52_pos < 30 else ('Mid range 🟡' if w52_pos < 70 else 'Near top 🔴')}</div>
  </div>
  <div style="flex:1;min-width:200px;background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 14px;">
    <div style="font-size:0.68rem;color:#aaa;text-transform:uppercase;margin-bottom:4px;">💰 Opportunity Analysis</div>
    <div style="font-size:0.82rem;color:rgba(255,255,255,0.9);line-height:1.6;">
      {opp_text}<br>{if_buy_html}
    </div>
  </div>
</div>
</div>"""


# ─────────────────────────── Prediction card renderer ──────────────────────


def render_pred_card(pred_result: dict, cur_price: float) -> str:
    pred = pred_result.get("prediction", {})
    quant = pred_result.get("quant_prediction", {})
    sentiment = pred_result.get("sentiment_report", "No sentiment data")
    cs_pats = pred_result.get("candlestick_patterns", [])
    ch_pats = pred_result.get("chart_patterns", [])
    
    direction = pred.get("direction", "NEUTRAL")
    quant_dir = quant.get("direction", direction)
    
    # Extract just the first line of sentiment for the badge
    sent_lines = sentiment.strip().split('\n')
    sent_summary = sent_lines[2] if len(sent_lines) > 2 else "Neutral"
    if "Overall View:" in sent_summary:
        sent_summary = sent_summary.replace("Overall View: ", "").split("—")[0].strip()
    
    conf = pred.get("confidence", 50)
    target = pred.get("price_target", cur_price)
    tgt_pct = pred.get("price_target_pct", 0)
    pattern = quant.get("primary_pattern", "—")
    reasoning = pred.get("reasoning", "")
    sl = pred.get("stop_loss")
    tf = pred.get("timeframe", "3-5 days")
    risk = pred.get("risk_level", "MEDIUM")
    vol_conf = pred.get("volume_confirmation", False)

    card_cls = {"BULLISH": "pred-bull", "BEARISH": "pred-bear"}.get(
        direction, "pred-neutral"
    )
    icon = {"BULLISH": "🟢 ▲", "BEARISH": "🔴 ▼"}.get(direction, "🟡 ◆")
    sign = "+" if tgt_pct >= 0 else ""

    # Use detected_patterns from quant engine (with trend-adjusted confidence)
    detected_patterns = pred.get("detected_patterns", []) or quant.get("detected_patterns", [])
    
    # Build pattern pills from the raw candlestick/chart patterns (for backwards compat)
    all_pats = (cs_pats + ch_pats)[:4]
    pills = " ".join(_ppill(p) for p in all_pats)
    
    # Build detailed pattern list from detected_patterns strings
    pattern_details_html = ""
    if detected_patterns:
        pat_items = "".join(
            f'<span style="display:inline-block;background:rgba(255,255,255,0.08);padding:3px 8px;border-radius:4px;margin:2px;font-size:0.75rem;">{p}</span>'
            for p in detected_patterns[:6]
        )
        pattern_details_html = f'''<div style="margin-top:8px;">
<div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;margin-bottom:4px;">📊 Detected Patterns (Trend-Adjusted)</div>
<div style="display:flex;flex-wrap:wrap;gap:2px;">{pat_items}</div>
</div>'''

    # Trend context badge
    trend = quant.get("trend", "sideways")
    trend_icons = {"uptrend": "📈 Uptrend", "downtrend": "📉 Downtrend", "sideways": "↔️ Sideways"}
    trend_label = trend_icons.get(trend, "↔️ Sideways")

    sl_html = (
        f'<div class="pred-metric"><span class="pred-metric-label">Stop Loss</span>₹{sl:,.2f}</div>'
        if sl
        else ""
    )
    vol_html = (
        '<div class="pred-metric"><span class="pred-metric-label">Volume</span>✅ Confirmed</div>'
        if vol_conf
        else ""
    )
    sup = pred.get("support_level", cur_price * 0.97)
    res = pred.get("resistance_level", cur_price * 1.03)
    
    # Quant confidence for comparison
    quant_conf = quant.get("confidence", 50)

    return f"""
<div class="{card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:0.8rem;opacity:0.8;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;color:#FFD700;">🤖 Multi-Agent Orchestrator Verdict</div>
      <div class="pred-direction">{icon} {direction}</div>
      <div class="pred-pattern">📊 Primary: {pattern} · {trend_label}</div>
    </div>
    <div style="text-align:right;">
      <div class="pred-target">🎯 ₹{target:,.2f}</div>
      <div class="pred-conf">{sign}{tgt_pct:.2f}% &nbsp;·&nbsp; {tf}</div>
      <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%"></div></div>
      <div class="pred-conf" style="margin-top:3px;">Confidence: {conf}%</div>
    </div>
  </div>
  <div class="pred-metrics">
    {sl_html}
    <div class="pred-metric"><span class="pred-metric-label">Risk</span>{risk}</div>
    <div class="pred-metric"><span class="pred-metric-label">Support</span>₹{sup:,.2f}</div>
    <div class="pred-metric"><span class="pred-metric-label">Resistance</span>₹{res:,.2f}</div>
    {vol_html}
  </div>
  <div class="pred-reasoning">💡 <b>Orchestrator Reasoning:</b> {reasoning}</div>
  {pattern_details_html}
  <div style="margin-top:12px;display:flex;gap:10px;background:rgba(0,0,0,0.2);padding:10px;border-radius:8px;">
    <div style="flex:1;border-right:1px solid rgba(255,255,255,0.1);padding-right:10px;">
        <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;">Agent 1: Quantitative</div>
        <div style="font-size:0.85rem;font-weight:bold;margin-top:2px;">{quant_dir} ({quant_conf}%)</div>
    </div>
    <div style="flex:1;">
        <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;">Agent 2: Sentiment</div>
        <div style="font-size:0.85rem;font-weight:bold;margin-top:2px;">{sent_summary}</div>
    </div>
  </div>
  <div style="margin-top:10px;">{pills}</div>
</div>
"""




# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — MARKET RESEARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        '<p class="section-title">Market Research Agent</p>', unsafe_allow_html=True
    )
    st.caption(
        "Performs 5-6+ deep searches and writes a comprehensive 9-section Indian market report."
    )

    example_topics = [
        "⚡ Quick Commerce",
        "🚗 Electric Vehicles",
        "📚 EdTech",
        "💸 UPI Payments",
        "🛍️ D2C Brands",
        "🍳 Cloud Kitchen",
        "🌾 Agritech",
        "🏥 Health Insurance",
    ]
    st.markdown("**Quick Topics:**")
    cols = st.columns(4)
    selected_topic = ""
    for i, topic in enumerate(example_topics):
        if cols[i % 4].button(topic, key=f"topic_{i}"):
            selected_topic = topic.split(" ", 1)[1]

    research_topic = st.text_input(
        "Or enter a custom topic:",
        value=selected_topic,
        placeholder="e.g., Fintech lending, ONDC, Gaming, Space tech...",
        key="research_topic_input",
    )
    research_btn = st.button("🚀 Start Research", type="primary", key="research_btn")

    if research_btn and research_topic:
        from agent import run_market_research_agent

        st.markdown("---")
        st.markdown("**Live Research Log:**")
        log_ph = st.empty()
        prog = st.progress(0.0, text="Initializing...")
        msgs = []

        def _status(msg):
            ts = datetime.now().strftime("%H:%M:%S")
            msgs.append(f"[{ts}] {msg}")
            log_ph.markdown(
                f'<div class="status-log">{"<br>".join(msgs[-8:])}</div>',
                unsafe_allow_html=True,
            )

        def _prog(v):
            prog.progress(min(v, 1.0), text=f"Researching... {int(v*100)}%")

        with st.spinner("Researching..."):
            try:
                report = run_market_research_agent(research_topic, _status, _prog)
                prog.progress(1.0, text="✅ Done!")
                st.success("Research complete!")
                st.markdown("---")
                st.markdown(
                    '<p class="section-title">Research Report</p>',
                    unsafe_allow_html=True,
                )
                r1, r2 = st.tabs(["📄 Formatted", "📝 Raw Markdown"])
                with r1:
                    st.markdown(report)
                with r2:
                    st.code(report, language="markdown")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"report_{research_topic.replace(' ','_').lower()[:40]}_{ts}.md"
                st.download_button(
                    "⬇️ Download Report (.md)",
                    data=f"# {research_topic}\n\n---\n\n{report}",
                    file_name=fn,
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Research failed: {e}")
    elif research_btn:
        st.warning("Please enter a topic.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}

    st.markdown(
        '<p class="section-title">📈 Portfolio Tracker + AI Prediction</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Candlestick charts · Bulkowski pattern detection · Groq AI price prediction · Live NSE/BSE prices"
    )

    # ── Add stock ─────────────────────────────────────────────────────────
    with st.expander("➕ Add New Stock", expanded=len(st.session_state.portfolio) == 0):
        def quick_add_cb(n, t):
            st.session_state["form_name"] = n
            st.session_state["form_ticker"] = t
            st.session_state["stock_search"] = "-- Type to search or select from list --"

        st.markdown("**Quick Add:**")
        qa_cols = st.columns(4)
        for i, (name, ticker) in enumerate(QUICK_ADD_STOCKS):
            qa_cols[i % 4].button(f"+ {name.split()[0]}", key=f"qa_{i}", on_click=quick_add_cb, args=(name, ticker))

        st.markdown("---")
        
        from nse_stocks import ALL_STOCKS_DEDUPED
        stock_options = ["-- Type to search or select from list --"] + sorted([f"{n} ──────── {t}" for t, n, _ in ALL_STOCKS_DEDUPED])
        
        def update_form_from_search():
            val = st.session_state.get("stock_search")
            if val and val != "-- Type to search or select from list --":
                parts = val.split(" ──────── ")
                if len(parts) == 2:
                    st.session_state["form_name"] = parts[0]
                    st.session_state["form_ticker"] = parts[1]

        sel_col, _ = st.columns([3, 1])
        with sel_col:
            st.selectbox(
                "🔍 Search NSE Stocks (Auto-fills below)", 
                stock_options, 
                key="stock_search", 
                on_change=update_form_from_search
            )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1])
        with c1:
            add_name = st.text_input(
                "Company Name",
                placeholder="Reliance Industries",
                key="form_name",
            )
        with c2:
            add_ticker = st.text_input(
                "NSE Ticker",
                placeholder="RELIANCE.NS",
                key="form_ticker",
            )
        with c3:
            add_qty = st.number_input("Qty", min_value=1, value=1, key="form_qty")
        with c4:
            add_price = st.number_input(
                "Avg Buy (₹)",
                min_value=0.01,
                value=100.0,
                format="%.2f",
                key="form_price",
            )

        if st.button("✅ Add", type="primary", key="add_btn"):
            if add_name and add_ticker:
                st.session_state.portfolio.append(
                    {
                        "id": f"{add_ticker}_{int(time.time())}",
                        "company": add_name,
                        "ticker": add_ticker.upper(),
                        "quantity": add_qty,
                        "avg_price": add_price,
                        "added_on": datetime.now().strftime("%Y-%m-%d"),
                    }
                )
                save_portfolio(st.session_state.portfolio)
                for k in ["form_name", "form_ticker", "stock_search"]:
                    st.session_state.pop(k, None)
                st.success(f"Added {add_name}!")
                st.rerun()
            else:
                st.warning("Fill in company name and ticker.")

    # ── Action buttons ────────────────────────────────────────────────────
    ca, cb, cc, _ = st.columns([1, 1.2, 1, 3])
    refresh_btn = ca.button("🔄 Refresh Prices", key="refresh_btn")
    predict_all = cb.button("🤖 Predict All Stocks", key="predict_all")
    export_btn = cc.button("📥 Export CSV", key="export_btn")

    if not st.session_state.portfolio:
        st.info("Portfolio is empty. Add stocks using the form above.")
        st.stop()

    # ── Migrate old portfolio entries missing required keys ───────────────
    for entry in st.session_state.portfolio:
        if "id" not in entry:
            entry["id"] = f"{entry.get('ticker','?')}_{int(time.time())}"
        if "company" not in entry:
            entry["company"] = entry.get("name", entry.get("ticker", "Unknown"))
        if "ticker" not in entry:
            entry["ticker"] = entry.get("symbol", "UNKNOWN")
        if "quantity" not in entry:
            entry["quantity"] = 1
        if "avg_price" not in entry:
            entry["avg_price"] = 0.0
        if "added_on" not in entry:
            entry["added_on"] = "2026-01-01"
    save_portfolio(st.session_state.portfolio)  # persist migrated data

    # ── Fetch live prices in BATCH ────────────────────────────────────────
    with st.spinner("Fetching live prices from NSE/BSE (Batch)..."):
        portfolio_json = json.dumps(st.session_state.portfolio)
        stock_data = fetch_portfolio_snapshots_cached(portfolio_json)

    # ── Summary metrics ───────────────────────────────────────────────────
    tot_inv = tot_cur = 0
    winners = losers = 0
    for entry in st.session_state.portfolio:
        snap = stock_data.get(entry["id"], {})
        if "current_price" in snap:
            iv = entry["quantity"] * entry["avg_price"]
            cv = entry["quantity"] * snap["current_price"]
            tot_inv += iv
            tot_cur += cv
            if cv >= iv:
                winners += 1
            else:
                losers += 1

    tot_pnl = tot_cur - tot_inv
    pnl_pct = (tot_pnl / tot_inv * 100) if tot_inv else 0

    st.markdown("---")
    st.markdown(
        '<p class="section-title">Portfolio Summary</p>', unsafe_allow_html=True
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Invested", f"₹{tot_inv:,.0f}")
    m2.metric("Current Value", f"₹{tot_cur:,.0f}")
    m3.metric("Total P&L", f"₹{tot_pnl:+,.0f}", delta=f"{pnl_pct:+.2f}%")
    m4.metric("Winners 🟢", str(winners))
    m5.metric("Losers 🔴", str(losers))

    # ── Holdings ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Holdings</p>', unsafe_allow_html=True)

    to_remove = []

    for entry in st.session_state.portfolio:
        snap = stock_data.get(entry["id"], {})
        has_data = "current_price" in snap
        df_stock = snap.get("df")
        pred_key = entry["id"]

        st.markdown(f"#### {entry['company']}  `{entry['ticker']}`")

        if not has_data:
            st.error(f"⚠️ {snap.get('error', 'Data unavailable')}")
            if st.button("🗑️ Remove", key=f"rm_{pred_key}"):
                to_remove.append(pred_key)
            st.divider()
            continue

        # ── Predict button & Timeframe Selector ───────────────────────
        period_key = f"period_{pred_key}"
        if period_key not in st.session_state:
            st.session_state[period_key] = "Medium Term (1-2 months)"

        timeframe_options = {
            "Swing Trade (1-3 days)": 14,     # ~2 weeks of data
            "Short Term (1-2 weeks)": 80,     # ~4 months of data
            "Medium Term (1-2 months)": 180,  # ~9 months of data
            "Long Term (4-6 months)": 500,    # ~2 years of data
        }

        tf_col, pb_col, _ = st.columns([2, 2, 2])
        selected_tf = tf_col.selectbox(
            "Analysis Timeframe",
            list(timeframe_options.keys()),
            index=list(timeframe_options.keys()).index(st.session_state[period_key]),
            key=f"sel_{period_key}",
            label_visibility="collapsed",
        )

        if selected_tf != st.session_state[period_key]:
            st.session_state[period_key] = selected_tf
            st.session_state.predictions.pop(
                pred_key, None
            )  # Clear old prediction on change
            st.rerun()

        run_pred = pb_col.button(
            f"🤖 Run AI Prediction", key=f"pred_{pred_key}", use_container_width=True
        )

        selected_tf = st.session_state[period_key]
        selected_days = timeframe_options[selected_tf]

        if selected_tf == "Swing Trade (1-3 days)":
            df_hourly = fetch_hourly_data_cached(entry["ticker"])
            if df_hourly is not None and not df_hourly.empty:
                # Roughly 7 trading hours per day
                df_sliced = df_hourly.tail(selected_days * 7)
                chart_display_days = len(df_sliced)
            else:
                df_sliced = df_stock.tail(selected_days) if df_stock is not None else None
                chart_display_days = selected_days
        else:
            df_sliced = df_stock.tail(selected_days) if df_stock is not None else None
            chart_display_days = selected_days

        if run_pred or predict_all:
            if df_sliced is not None and not df_sliced.empty:
                with st.spinner(
                    f"Running AI analysis ({st.session_state[period_key]}) for {entry['company']}..."
                ):
                    try:
                        from prediction_engine import run_full_prediction

                        result = run_full_prediction(
                            entry["company"],
                            entry["ticker"],
                            snap,
                            df_sliced,
                            term=st.session_state[period_key],
                        )
                        st.session_state.predictions[pred_key] = result
                    except Exception as e:
                        st.session_state.predictions[pred_key] = {"error": str(e)}

        # ── Prediction card ABOVE chart ───────────────────────────────
        pred_result = st.session_state.predictions.get(pred_key)
        if pred_result:
            if "error" in pred_result:
                st.warning(f"Prediction error: {pred_result['error']}")
            else:
                st.markdown(
                    render_pred_card(pred_result, snap["current_price"]),
                    unsafe_allow_html=True,
                )

        # ── Metrics + Chart ───────────────────────────────────────────
        col_m, col_c = st.columns([1, 2.2])

        with col_m:
            price = snap["current_price"]
            chg = snap["today_change"]
            inv_val = entry["quantity"] * entry["avg_price"]
            cur_val = entry["quantity"] * price
            pnl = cur_val - inv_val
            pnl_p = (pnl / inv_val * 100) if inv_val else 0
            pnl_cls = "profit" if pnl >= 0 else "loss"

            st.metric("Live Price", f"₹{price:,.2f}", delta=f"{chg:+.2f}%")
            ca2, cb2 = st.columns(2)
            ca2.metric("RSI", f"{snap['rsi']:.1f}" if snap.get("rsi") else "—")
            cb2.metric("Qty", str(entry["quantity"]))
            st.markdown(
                f"**P&L:** <span class='{pnl_cls}'>₹{pnl:+,.0f} ({pnl_p:+.2f}%)</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Avg buy: ₹{entry['avg_price']:,.2f}")
            ma_parts = []
            if snap.get("ma20"):
                ma_parts.append(f"MA20 ₹{snap['ma20']:,.2f}")
            if snap.get("ma50"):
                ma_parts.append(f"MA50 ₹{snap['ma50']:,.2f}")
            if ma_parts:
                st.caption(" · ".join(ma_parts))
            st.caption(f"52W: ₹{snap['week52_low']:,.2f} – ₹{snap['week52_high']:,.2f}")
            st.markdown(_signal_badge(snap["signal"]), unsafe_allow_html=True)

            # ── Extra metrics ────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if snap.get("return_1m") is not None:
                r1m_cls = "profit" if snap["return_1m"] >= 0 else "loss"
                st.markdown(
                    f"**1M Return:** <span class='{r1m_cls}'>{snap['return_1m']:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
            vol = snap.get("today_volume", 0)
            avg_vol = snap.get("avg_volume", 0)
            if vol and avg_vol:
                vol_r = vol / avg_vol
                vol_str = f"{vol/1e6:.1f}M" if vol > 1e6 else f"{vol/1e3:.0f}K"
                st.caption(f"Volume: {vol_str} ({vol_r:.1f}× avg)")

        with col_c:
            if df_sliced is not None and not df_sliced.empty:
                pred_for_chart = None
                if pred_result and "error" not in pred_result:
                    pred_for_chart = pred_result["prediction"]
                html_chart = build_tradingview_lightweight_chart(
                    df_sliced, entry["ticker"], pred_for_chart, days=chart_display_days
                )
                components.html(html_chart, height=420, scrolling=False)
            else:
                st.info("No chart data available.")

        # ── Smart Action Advisor ──────────────────────────────────────
        smart = compute_smart_action(entry, snap, pred_result)
        st.markdown(
            render_smart_action(entry, snap, smart),
            unsafe_allow_html=True,
        )

        # ── Edit / Remove buttons ─────────────────────────────────────
        btn_edit, btn_rm, _ = st.columns([1, 1, 4])
        edit_toggled = btn_edit.button("✏️ Edit", key=f"edit_{pred_key}")
        if btn_rm.button("🗑️ Remove", key=f"remove_{pred_key}"):
            to_remove.append(pred_key)

        # Toggle edit state
        edit_state_key = f"editing_{pred_key}"
        if edit_toggled:
            st.session_state[edit_state_key] = not st.session_state.get(edit_state_key, False)

        if st.session_state.get(edit_state_key, False):
            with st.container():
                st.markdown(
                    f"<div style='background:rgba(255,107,53,0.08);border:1px solid rgba(255,107,53,0.3);"
                    f"border-radius:10px;padding:16px 20px;margin:8px 0 16px;'>"
                    f"<span style='font-family:Syne,sans-serif;font-weight:700;font-size:0.9rem;"
                    f"color:#FF6B35;'>✏️ Edit {entry['company']}</span></div>",
                    unsafe_allow_html=True,
                )
                ec1, ec2, ec3 = st.columns([1, 1, 1])
                new_qty = ec1.number_input(
                    "Quantity",
                    min_value=1,
                    value=entry["quantity"],
                    key=f"eq_{pred_key}",
                )
                new_avg = ec2.number_input(
                    "Avg Buy Price (₹)",
                    min_value=0.01,
                    value=float(entry["avg_price"]),
                    format="%.2f",
                    key=f"eap_{pred_key}",
                )
                if ec3.button("💾 Save", key=f"save_{pred_key}", type="primary"):
                    # Update entry in session state
                    for e in st.session_state.portfolio:
                        if e["id"] == pred_key:
                            e["quantity"] = new_qty
                            e["avg_price"] = new_avg
                            break
                    save_portfolio(st.session_state.portfolio)
                    st.session_state[edit_state_key] = False
                    st.success(f"Updated {entry['company']}: Qty={new_qty}, Avg Buy=₹{new_avg:,.2f}")
                    st.rerun()

        st.divider()

    # ── Process removals ──────────────────────────────────────────────────
    if to_remove:
        st.session_state.portfolio = [
            e for e in st.session_state.portfolio if e["id"] not in to_remove
        ]
        save_portfolio(st.session_state.portfolio)
        st.rerun()

    # ── Export CSV ────────────────────────────────────────────────────────
    if export_btn:
        rows = []
        for entry in st.session_state.portfolio:
            snap = stock_data.get(entry["id"], {})
            if "current_price" in snap:
                iv = entry["quantity"] * entry["avg_price"]
                cv = entry["quantity"] * snap["current_price"]
                pnl = cv - iv
                pr = st.session_state.predictions.get(entry["id"])
                pd_ = pr["prediction"] if pr and "error" not in pr else {}
                rows.append(
                    {
                        "Company": entry["company"],
                        "Ticker": entry["ticker"],
                        "Qty": entry["quantity"],
                        "Avg Buy (₹)": entry["avg_price"],
                        "Live (₹)": snap["current_price"],
                        "Today %": snap["today_change"],
                        "RSI": snap.get("rsi", ""),
                        "MA20": snap.get("ma20", ""),
                        "MA50": snap.get("ma50", ""),
                        "Signal": snap.get("signal", ""),
                        "AI Direction": pd_.get("direction", ""),
                        "AI Target (₹)": pd_.get("price_target", ""),
                        "AI Confidence": pd_.get("confidence", ""),
                        "Pattern": pd_.get("primary_pattern", ""),
                        "Invested (₹)": round(iv, 2),
                        "Current (₹)": round(cv, 2),
                        "P&L (₹)": round(pnl, 2),
                        "P&L %": round((pnl / iv * 100) if iv else 0, 2),
                    }
                )
        if rows:
            df_exp = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Download CSV",
                data=df_exp.to_csv(index=False),
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — LIVE TICKER BOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    if "board_pred" not in st.session_state:
        st.session_state.board_pred = {}

    st.markdown(
        '<p class="section-title">📊 Live Ticker Board</p>', unsafe_allow_html=True
    )
    st.caption(
        "Live MCX-style green/red cards for your portfolio • Click Predict for AI analysis"
    )

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks in the Portfolio Tracker tab.")
        st.stop()

    # ── Controls ─────────────────────────────────────────────────────────
    bc1, bc2, _ = st.columns([1, 1.5, 4])
    board_refresh = bc1.button("🔄 Refresh", key="board_refresh")
    board_pred_all = bc2.button("🤖 Predict All", key="board_pred_all")

    # ── Fetch data in BATCH ───────────────────────────────────────────────
    with st.spinner("Fetching live prices (Batch)..."):
        portfolio_json = json.dumps(st.session_state.portfolio)
        board_data = fetch_portfolio_snapshots_cached(portfolio_json)

    # ── Predict all if requested ──────────────────────────────────────────
    if board_pred_all:
        with st.spinner("Running AI predictions for all stocks..."):
            for entry in st.session_state.portfolio:
                eid = entry.get("id", entry.get("ticker"))
                snap = board_data.get(eid, {})
                df_s = snap.get("df")
                if df_s is not None and not df_s.empty and "current_price" in snap:
                    try:
                        from prediction_engine import run_full_prediction

                        result = run_full_prediction(
                            entry.get("company", "Unknown"),
                            entry.get("ticker", "UNKNOWN"),
                            snap,
                            df_s,
                        )
                        st.session_state.board_pred[eid] = result
                    except Exception as e:
                        st.session_state.board_pred[eid] = {"error": str(e)}

    # ── Separate into gainers and losers ─────────────────────────────────
    gainers = []
    losers = []

    for entry in st.session_state.portfolio:
        eid = entry.get("id", entry.get("ticker", "?"))
        snap = board_data.get(eid, {})
        if "current_price" not in snap:
            continue

        cur = snap["current_price"]
        avg = entry.get("avg_price", 0)
        qty = entry.get("quantity", 1)
        invested = avg * qty
        current = cur * qty
        pnl = current - invested
        pnl_pct = (pnl / invested * 100) if invested else 0
        today = snap.get("today_change", 0)

        card = {
            "id": eid,
            "entry": entry,
            "snap": snap,
            "cur": cur,
            "today": today,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "invested": invested,
            "current": current,
        }
        if pnl >= 0:
            gainers.append(card)
        else:
            losers.append(card)

    # Sort by P&L %
    gainers.sort(key=lambda x: x["pnl_pct"], reverse=True)
    losers.sort(key=lambda x: x["pnl_pct"])

    # ── Summary bar ───────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Gainers", str(len(gainers)))
    s2.metric("🔴 Losers", str(len(losers)))
    tot_pnl = sum(c["pnl"] for c in gainers + losers)
    tot_inv = sum(c["invested"] for c in gainers + losers)
    s3.metric(
        "Total P&L",
        f"₹{tot_pnl:+,.0f}",
        delta=f"{(tot_pnl/tot_inv*100) if tot_inv else 0:+.2f}%",
    )
    s4.metric("Total Stocks", str(len(gainers) + len(losers)))

    st.markdown("---")

    # ── Helper: build one ticker card HTML ───────────────────────────────
    def _ticker_card_html(card: dict, is_green: bool) -> str:
        entry = card["entry"]
        snap = card["snap"]
        name = entry.get("company", "Unknown")
        ticker = entry.get("ticker", "—")
        cur = card["cur"]
        today = card["today"]
        pnl = card["pnl"]
        pnl_pct = card["pnl_pct"]

        cls = "tc-green" if is_green else "tc-red"
        arrow = (
            '<span class="tc-arrow-up">▲</span>'
            if today >= 0
            else '<span class="tc-arrow-down">▼</span>'
        )
        pct_cls = "tc-chg-pct-g" if today >= 0 else "tc-chg-pct-r"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_p_s = "+" if pnl_pct >= 0 else ""

        # Shorten name for card
        short_name = name if len(name) <= 18 else name[:17] + "…"

        return f"""
<div class="{cls}">
  <div class="tc-name">{short_name}</div>
  <div class="tc-ticker-tag">{ticker}</div>
  <div class="tc-price">₹{cur:,.2f}</div>
  <div class="tc-change-row">
    {arrow}
    <span class="tc-chg-val">{abs(today):.2f}</span>
    <span class="{pct_cls}">&nbsp;{today:+.2f}%</span>
  </div>
  <div class="tc-pnl-row">
    P&amp;L: {pnl_sign}₹{pnl:,.0f} &nbsp;|&nbsp; {pnl_p_s}{pnl_pct:.2f}%
  </div>
</div>"""

    # ── GAINERS SECTION ───────────────────────────────────────────────────
    if gainers:
        st.markdown(
            f'<div class="board-hdr-g">🟢 GAINERS ({len(gainers)} stocks)</div>',
            unsafe_allow_html=True,
        )

        # Render cards in rows of 4
        for row_start in range(0, len(gainers), 4):
            row_cards = gainers[row_start : row_start + 4]
            cols = st.columns(len(row_cards))
            for i, card in enumerate(row_cards):
                with cols[i]:
                    st.markdown(
                        _ticker_card_html(card, is_green=True), unsafe_allow_html=True
                    )

                    eid = card["id"]
                    pred_res = st.session_state.board_pred.get(eid)

                    # Predict button
                    if st.button(
                        "🤖 Predict", key=f"bpred_{eid}", use_container_width=True
                    ):
                        snap = card["snap"]
                        df_s = snap.get("df")
                        if df_s is not None and not df_s.empty:
                            with st.spinner("Analyzing..."):
                                try:
                                    from prediction_engine import run_full_prediction

                                    result = run_full_prediction(
                                        card["entry"].get("company", "Unknown"),
                                        card["entry"].get("ticker", "UNKNOWN"),
                                        snap,
                                        df_s,
                                    )
                                    st.session_state.board_pred[eid] = result
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    # Show mini prediction result below card
                    if pred_res and "error" not in pred_res:
                        pred = pred_res["prediction"]
                        dir_ = pred.get("direction", "NEUTRAL")
                        conf = pred.get("confidence", 0)
                        tgt = pred.get("price_target", 0)
                        pat = pred.get("primary_pattern", "—")
                        d_color = {"BULLISH": "#00ff88", "BEARISH": "#ff4444"}.get(
                            dir_, "#ffd700"
                        )
                        st.markdown(
                            f"""
<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;
     margin-top:4px;border-left:3px solid {d_color};font-size:.78rem;color:white;">
  <b style="color:{d_color}">{dir_}</b> · ₹{tgt:,.0f} · {conf}%<br>
  <span style="opacity:.7">{pat}</span>
</div>""",
                            unsafe_allow_html=True,
                        )
                    elif pred_res and "error" in pred_res:
                        st.caption(f"⚠️ {pred_res['error'][:40]}")

    else:
        st.markdown(
            '<div class="board-hdr-g">🟢 GAINERS — None today</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOSERS SECTION ────────────────────────────────────────────────────
    if losers:
        st.markdown(
            f'<div class="board-hdr-r">🔴 LOSERS ({len(losers)} stocks)</div>',
            unsafe_allow_html=True,
        )

        for row_start in range(0, len(losers), 4):
            row_cards = losers[row_start : row_start + 4]
            cols = st.columns(len(row_cards))
            for i, card in enumerate(row_cards):
                with cols[i]:
                    st.markdown(
                        _ticker_card_html(card, is_green=False), unsafe_allow_html=True
                    )

                    eid = card["id"]
                    pred_res = st.session_state.board_pred.get(eid)

                    if st.button(
                        "🤖 Predict", key=f"bpred_{eid}", use_container_width=True
                    ):
                        snap = card["snap"]
                        df_s = snap.get("df")
                        if df_s is not None and not df_s.empty:
                            with st.spinner("Analyzing..."):
                                try:
                                    from prediction_engine import run_full_prediction

                                    result = run_full_prediction(
                                        card["entry"].get("company", "Unknown"),
                                        card["entry"].get("ticker", "UNKNOWN"),
                                        snap,
                                        df_s,
                                    )
                                    st.session_state.board_pred[eid] = result
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    if pred_res and "error" not in pred_res:
                        pred = pred_res["prediction"]
                        dir_ = pred.get("direction", "NEUTRAL")
                        conf = pred.get("confidence", 0)
                        tgt = pred.get("price_target", 0)
                        pat = pred.get("primary_pattern", "—")
                        d_color = {"BULLISH": "#00ff88", "BEARISH": "#ff4444"}.get(
                            dir_, "#ffd700"
                        )
                        st.markdown(
                            f"""
<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;
     margin-top:4px;border-left:3px solid {d_color};font-size:.78rem;color:white;">
  <b style="color:{d_color}">{dir_}</b> · ₹{tgt:,.0f} · {conf}%<br>
  <span style="opacity:.7">{pat}</span>
</div>""",
                            unsafe_allow_html=True,
                        )
                    elif pred_res and "error" in pred_res:
                        st.caption(f"⚠️ {pred_res['error'][:40]}")
    else:
        st.markdown(
            '<div class="board-hdr-r">🔴 LOSERS — None today</div>',
            unsafe_allow_html=True,
        )

    # ── Stocks with no data ───────────────────────────────────────────────
    no_data = [
        e
        for e in st.session_state.portfolio
        if "current_price" not in board_data.get(e.get("id", e.get("ticker", "")), {})
    ]
    if no_data:
        st.markdown("---")
        st.caption(f"⚠️ No data: {', '.join(e.get('ticker','?') for e in no_data)}")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — NSE MARKET WATCH  (real-time, auto-refresh like Zerodha)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    from nse_stocks import NSE_STOCKS, NIFTY50, NIFTY50_NAMES

    st.markdown(
        '<p class="section-title">🌐 NSE Market Watch</p>', unsafe_allow_html=True
    )
    st.caption(
        "Real-time NSE prices • Auto-refreshes like Zerodha • No page reload needed"
    )

    # ── Static controls (outside fragment — never reset on refresh) ───────
    f1, f2, f3, f4 = st.columns([1.8, 1.5, 1.2, 1])

    universe_choice = f1.selectbox(
        "Index / Universe",
        [
            "⭐ Nifty 50",
            "💯 Nifty 100",
            "🌐 Nifty 500",
            "🏦 Banking & Finance",
            "💻 IT & Technology",
            "🚗 Automobile & EV",
            "⚡ Energy & Power",
            "🧪 Chemicals & Pharma",
            "🛒 FMCG & Consumer",
            "🏥 Healthcare & Diagnostics",
            "🏗️ Infrastructure & Real Estate",
            "🔩 Metals & Mining",
            "🏭 Manufacturing & Industrials",
            "📡 Telecom & Media",
            "📦 Logistics & Transport",
            "🛍️ Retail & D2C",
        ],
        key="mw_universe",
    )
    filter_move = f2.selectbox(
        "Show",
        ["All Stocks", "🟢 Gainers Only", "🔴 Losers Only", "🔥 Top Movers (>2%)"],
        key="mw_filter",
    )
    search_query = f3.text_input("Search", placeholder="e.g. Reliance", key="mw_search")
    sort_by = f4.selectbox("Sort by", ["Change %", "Price", "Name"], key="mw_sort")

    rb1, rb2, rb3, rb4 = st.columns([1, 1.2, 1.2, 4])
    load_btn = rb1.button("🚀 Load", type="primary", key="mw_load")
    interval_secs = rb2.selectbox(
        "Auto-refresh",
        [15, 30, 60, 120, 300],
        index=1,
        key="mw_interval",
        format_func=lambda x: f"Every {x}s",
    )
    live_toggle = rb3.toggle("⚡ Live Mode", value=False, key="mw_live")
    if live_toggle:
        rb4.markdown(
            '<span style="color:#00ff88;font-weight:700;font-size:.9rem">'
            + "🔴 LIVE — prices auto-updating</span>",
            unsafe_allow_html=True,
        )

    # ── Session state ─────────────────────────────────────────────────────
    for _k, _v in [
        ("mw_history", {}),
        ("mw_base", []),
        ("mw_loaded", False),
        ("mw_pred", {}),
        ("mw_universe_c", ""),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    def _build_tl(choice):
        if "Nifty 500" in choice:
            st.info(
                "⏳ Fetching live Nifty 500 constituents — this may take a moment..."
            )
            from nse_stocks import fetch_nifty500

            n500 = fetch_nifty500()
            return [(t, n, "Nifty 500") for t, n in n500]
        if "Nifty 100" in choice:
            from nse_stocks import NIFTY100, NIFTY100_NAMES

            return [
                (t, NIFTY100_NAMES.get(t, t.replace(".NS", "")), "Nifty 100")
                for t in NIFTY100
            ]
        if "Nifty 50" in choice:
            return [
                (t, NIFTY50_NAMES.get(t, t.replace(".NS", "")), "Nifty 50")
                for t in NIFTY50
            ]
        for sk, stocks in NSE_STOCKS.items():
            if sk in choice or choice in sk:
                return [(f"{s}.NS", n, sk) for s, n in stocks]
        return []

    # ── Full history load — ONLY on explicit Load click or universe change ─
    # live_toggle change must NOT trigger a reload
    universe_changed = universe_choice != st.session_state.mw_universe_c
    should_load = (
        load_btn
        or (not st.session_state.mw_loaded and not live_toggle)
        or (universe_changed and st.session_state.mw_loaded)
    )

    if should_load:
        tl = _build_tl(universe_choice)
        st.session_state.mw_universe_c = universe_choice
        prog = st.progress(0.0, text=f"Loading {len(tl)} stocks (history, RSI, 52W)...")
        history, base = {}, []
        
        tickers_list = [item[0] for item in tl]
        
        try:
            raw_data = yf.download(
                tickers_list,
                start="2026-01-01",
                group_by="ticker",
                progress=False,
                threads=True
            )
        except Exception as e:
            st.error(f"Failed to fetch market data: {e}")
            raw_data = None
            
        for idx, (sym, name, sector) in enumerate(tl):
            prog.progress((idx + 1) / len(tl), text=f"{sym} ({idx+1}/{len(tl)})")
            try:
                if raw_data is None or raw_data.empty:
                    continue
                
                # Extract this ticker's df
                if len(tickers_list) == 1:
                    df_t = raw_data
                else:
                    if hasattr(raw_data.columns, 'levels') and sym in raw_data.columns.levels[0]:
                        df_t = raw_data[sym]
                    elif hasattr(raw_data.columns, 'levels') and sym in raw_data.columns.get_level_values(0):
                        df_t = raw_data[sym]
                    else:
                        continue

                if df_t is None or df_t.empty or len(df_t) < 2:
                    continue
                df_t = df_t.dropna(subset=["Close"])
                if len(df_t) < 2:
                    continue
                
                rsi = None
                if len(df_t) >= 15:
                    try:
                        d = df_t["Close"].diff()
                        g = d.clip(lower=0).rolling(14).mean()
                        l = (-d.clip(upper=0)).rolling(14).mean()
                        r = g / l.replace(0, float("nan"))
                        v = (100 - 100 / (1 + r)).iloc[-1]
                        rsi = round(float(v), 1) if not np.isnan(v) else None
                    except:
                        pass
                
                vol = (
                    int(df_t["Volume"].iloc[-1])
                    if not np.isnan(df_t["Volume"].iloc[-1])
                    else 0
                )
                prev_c = float(df_t["Close"].iloc[-2])  # yesterday's close for % calc
                history[sym] = {
                    "df": df_t,
                    "prev_close": prev_c,
                    "hi52": round(float(df_t["High"].max()), 2),
                    "lo52": round(float(df_t["Low"].min()), 2),
                    "rsi": rsi,
                    "volume": vol,
                }
                base.append({"ticker": sym, "name": name, "sector": sector})
            except:
                continue
        prog.progress(
            1.0,
            text=f"✅ {len(base)} stocks loaded. Toggle Live Mode to auto-refresh prices.",
        )
        st.session_state.mw_history = history
        st.session_state.mw_base = base
        st.session_state.mw_loaded = True

    # ── LIVE FRAGMENT ─────────────────────────────────────────────────────
    # @st.fragment(run_every=N) re-runs ONLY this block every N seconds.
    # Uses a single yf.download(period="1d") batch call for ALL tickers at once
    # — one HTTP request instead of 50 sequential fast_info calls. Much faster.
    @st.fragment(run_every=(interval_secs if live_toggle else None))
    def _live_watch():
        history = st.session_state.mw_history
        base = st.session_state.mw_base
        s_query = st.session_state.get("mw_search", "")
        s_filter = st.session_state.get("mw_filter", "All Stocks")
        s_sort = st.session_state.get("mw_sort", "Change %")

        if not base:
            st.info("Click 🚀 Load to fetch market data first.")
            return

        now_str = datetime.now().strftime("%H:%M:%S")

        # ── BATCH price fetch — single API call for all tickers ───────
        # yf.download(period="2d") gets today + yesterday in one request
        tickers_str = " ".join(item["ticker"] for item in base)
        live_prices = {}  # ticker -> (current_price, prev_close)

        try:
            raw = yf.download(
                tickers_str,
                period="2d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            # Parse the multi-ticker DataFrame
            # Structure: columns are MultiIndex (field, ticker) or (ticker, field)
            for item in base:
                sym = item["ticker"]
                try:
                    # Try (ticker, field) first
                    if hasattr(raw.columns, "levels"):
                        lvl0 = raw.columns.get_level_values(0).unique().tolist()
                        lvl1 = raw.columns.get_level_values(1).unique().tolist()
                        # Detect structure
                        if sym in lvl0:
                            df_s = raw[sym]
                        elif sym in lvl1:
                            df_s = raw.xs(sym, level=1, axis=1)
                        else:
                            raise KeyError(sym)
                    else:
                        df_s = raw  # single ticker

                    df_s = df_s.dropna(subset=["Close"])
                    if len(df_s) >= 2:
                        cur = float(df_s["Close"].iloc[-1])
                        prev = float(df_s["Close"].iloc[-2])
                        if not np.isnan(cur) and cur > 0:
                            live_prices[sym] = (cur, prev)
                except Exception:
                    pass

        except Exception:
            pass

        # ── Build rows — merge batch prices with cached history ────────
        rows = []
        for item in base:
            sym = item["ticker"]
            hist = history.get(sym, {})
            if not hist:
                continue

            hist_prev = hist["prev_close"]

            if sym in live_prices:
                cur, prev = live_prices[sym]
            else:
                # Fallback: fast_info for this single stock only
                try:
                    fi = yf.Ticker(sym).fast_info
                    lp = float(fi.last_price)
                    pc = float(fi.previous_close) if fi.previous_close else hist_prev
                    if np.isnan(lp) or lp <= 0:
                        raise ValueError()
                    cur, prev = lp, (pc if not np.isnan(pc) else hist_prev)
                except:
                    cur, prev = hist_prev, hist_prev

            if prev == 0 or np.isnan(cur):
                continue
            chg = (cur - prev) / prev * 100
            rows.append(
                {
                    "ticker": sym,
                    "name": item["name"],
                    "sector": item["sector"],
                    "price": round(cur, 2),
                    "chg_abs": round(cur - prev, 2),
                    "chg_pct": round(chg, 2),
                    "volume": hist.get("volume", 0),
                    "hi52": hist.get("hi52", 0),
                    "lo52": hist.get("lo52", 0),
                    "rsi": hist.get("rsi"),
                }
            )

        # filters
        data = list(rows)
        if s_query:
            q = s_query.lower()
            data = [
                d for d in data if q in d["name"].lower() or q in d["ticker"].lower()
            ]
        if "Gainers" in s_filter:
            data = [d for d in data if d["chg_pct"] > 0]
        elif "Losers" in s_filter:
            data = [d for d in data if d["chg_pct"] < 0]
        elif "Top Movers" in s_filter:
            data = [d for d in data if abs(d["chg_pct"]) >= 2]
        if s_sort == "Change %":
            data.sort(key=lambda x: x["chg_pct"], reverse=True)
        elif s_sort == "Price":
            data.sort(key=lambda x: x["price"], reverse=True)
        elif s_sort == "Name":
            data.sort(key=lambda x: x["name"])

        if not data:
            st.info("No stocks match filter.")
            return

        gainers_c = sum(1 for d in data if d["chg_pct"] > 0)
        losers_c = sum(1 for d in data if d["chg_pct"] < 0)
        unch_c = len(data) - gainers_c - losers_c
        top_g = max(data, key=lambda x: x["chg_pct"])
        top_l = min(data, key=lambda x: x["chg_pct"])

        pulse = "#00ff88" if live_toggle else "#aaa"
        anim = "animation:pulse 1s infinite;" if live_toggle else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
            + f'<span style="width:10px;height:10px;border-radius:50%;background:{pulse};display:inline-block;{anim}"></span>'
            + f'<span style="color:#aaa;font-size:.8rem;">Updated: {now_str} IST &nbsp;·&nbsp; {len(data)} stocks &nbsp;·&nbsp; '
            + (
                f"Auto-refreshing every {interval_secs}s"
                if live_toggle
                else "Manual mode — enable Live Mode"
            )
            + "</span></div>",
            unsafe_allow_html=True,
        )

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.markdown(
            f'<div class="mw-stat-g"><div class="mw-stat-val">{gainers_c}</div><div class="mw-stat-lbl">🟢 Advancing</div></div>',
            unsafe_allow_html=True,
        )
        sc2.markdown(
            f'<div class="mw-stat-r"><div class="mw-stat-val">{losers_c}</div><div class="mw-stat-lbl">🔴 Declining</div></div>',
            unsafe_allow_html=True,
        )
        sc3.markdown(
            f'<div class="mw-stat-n"><div class="mw-stat-val">{unch_c}</div><div class="mw-stat-lbl">⬜ Unchanged</div></div>',
            unsafe_allow_html=True,
        )
        sc4.markdown(
            f'<div class="mw-stat-g"><div class="mw-stat-val" style="font-size:.9rem">{top_g["name"][:14]}</div><div class="mw-stat-lbl">🏆 Top Gainer +{top_g["chg_pct"]:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
        sc5.markdown(
            f'<div class="mw-stat-r"><div class="mw-stat-val" style="font-size:.9rem">{top_l["name"][:14]}</div><div class="mw-stat-lbl">📉 Top Loser {top_l["chg_pct"]:.2f}%</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """<div class="mw-header">
  <span style="min-width:175px">Company</span>
  <span style="min-width:105px">Symbol</span>
  <span style="min-width:100px;text-align:right">Price (₹)</span>
  <span style="min-width:85px;text-align:right">Change ₹</span>
  <span style="min-width:80px;text-align:right">Chg %</span>
  <span style="min-width:55px;text-align:right">RSI</span>
  <span style="min-width:85px;text-align:right">Volume</span>
  <span style="min-width:140px;text-align:right">Sector</span>
  <span style="min-width:80px;text-align:center">Actions</span>
</div>""",
            unsafe_allow_html=True,
        )

        for row in data:
            is_up = row["chg_pct"] >= 0
            row_cls = "mw-green" if is_up else "mw-red"
            chg_cls = "mw-chg-g" if is_up else "mw-chg-r"
            arrow = "▲" if is_up else "▼"
            rsi_val = row.get("rsi")
            rsi_str = f"{rsi_val}" if rsi_val else "—"
            rsi_color = (
                "#ff6666"
                if (rsi_val and rsi_val > 70)
                else ("#00ff88" if (rsi_val and rsi_val < 30) else "#aaa")
            )
            vol = row["volume"]
            vol_str = (
                f"{vol/1e6:.1f}M"
                if vol > 1e6
                else (f"{vol/1e3:.0f}K" if vol > 1e3 else str(vol))
            )
            sec_s = (
                row["sector"].split(" ", 1)[1][:18]
                if " " in row["sector"]
                else row["sector"][:18]
            )

            col_row, col_pred, col_add = st.columns([9.5, 0.7, 0.8])
            with col_row:
                st.markdown(
                    f"""<div class="mw-ticker-row {row_cls}">
  <span class="mw-name">{row["name"][:24]}</span>
  <span class="mw-sym">{row["ticker"].replace(".NS","")}</span>
  <span class="mw-price">₹{row["price"]:,.2f}</span>
  <span class="{chg_cls}">{arrow} {abs(row["chg_abs"]):.2f}</span>
  <span class="{chg_cls}">{row["chg_pct"]:+.2f}%</span>
  <span style="min-width:55px;text-align:right;font-size:.82rem;font-weight:700;color:{rsi_color}">{rsi_str}</span>
  <span class="mw-vol">{vol_str}</span>
  <span class="mw-sector">{sec_s}</span>
</div>""",
                    unsafe_allow_html=True,
                )

            sc = row["ticker"].replace(".", "_").replace("-", "_")
            with col_pred:
                if st.button("🤖", key=f"mwp_{sc}", help=f"AI Predict {row['name']}"):
                    with st.spinner(f"Analyzing {row['name']}..."):
                        try:
                            from prediction_engine import run_full_prediction

                            df_t = history.get(row["ticker"], {}).get("df")
                            if df_t is not None and not df_t.empty:
                                result = run_full_prediction(
                                    row["name"],
                                    row["ticker"],
                                    {
                                        "current_price": row["price"],
                                        "today_change": row["chg_pct"],
                                        "rsi": row.get("rsi"),
                                        "ma20": None,
                                        "ma50": None,
                                        "week52_high": row["hi52"],
                                        "week52_low": row["lo52"],
                                        "return_1m": None,
                                        "avg_volume": row["volume"],
                                        "today_volume": row["volume"],
                                    },
                                    df_t,
                                )
                                st.session_state.mw_pred[row["ticker"]] = result
                        except Exception as e:
                            st.session_state.mw_pred[row["ticker"]] = {"error": str(e)}

            with col_add:
                is_in_portfolio = any(
                    p.get("ticker") == row["ticker"] for p in st.session_state.portfolio
                ) 
                if is_in_portfolio:
                    st.button(
                        "✅",
                        key=f"mwa_in_{sc}",
                        help=f"{row['name']} is already in Portfolio",
                        disabled=True,
                    )
                else:
                    if st.button(
                        "➕",
                        key=f"mwa_add_{sc}",
                        help=f"Add {row['name']} to Portfolio",
                    ):
                        new_entry = {
                            "id": f"{row['ticker']}_{int(time.time())}",
                            "company": row["name"],
                            "ticker": row["ticker"],
                            "quantity": 1,
                            "avg_price": row["price"],
                            "added_on": datetime.now().strftime("%Y-%m-%d"),
                        }
                        st.session_state.portfolio.append(new_entry)
                        save_portfolio(st.session_state.portfolio)
                        st.toast(f"✅ Added {row['name']} to Portfolio Tracker!")
                        st.rerun()

            pr = st.session_state.mw_pred.get(row["ticker"])
            if pr and "error" not in pr:
                p = pr["prediction"]
                d_ = p.get("direction", "NEUTRAL")
                dc = {"BULLISH": "#00ff88", "BEARISH": "#ff4444"}.get(d_, "#ffd700")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.04);border-radius:8px;padding:10px 16px;'
                    + f'margin:0 0 8px 60px;border-left:3px solid {dc};font-size:.8rem;color:white;">'
                    + f'<b style="color:{dc}">{d_}</b> · 🎯 ₹{p.get("price_target",0):,.2f} · {p.get("confidence",0)}% · {p.get("timeframe","")}'
                    + f'<br><span style="opacity:.75">📊 {p.get("primary_pattern","—")} · {p.get("reasoning","")[:110]}</span></div>',
                    unsafe_allow_html=True,
                )
            elif pr and "error" in pr:
                st.caption(f"⚠️ {pr['error'][:60]}")

        st.markdown("---")
        live_label = (
            f"🔴 Auto-refreshing every {interval_secs}s"
            if live_toggle
            else "⚪ Enable Live Mode for auto-refresh"
        )
        st.caption(f"NSE via yfinance fast_info · {live_label}")

    _live_watch()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — SMART RECOMMENDATIONS  (AI Stock Screener + Chatbot)
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    from stock_scanner import run_scanner, chat_with_advisor

    st.markdown(
        '<p class="section-title">🎯 Smart Stock Recommendations</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "AI-powered scanner • Bulkowski pattern scoring • Top Buy & Sell picks • Groq chatbot for Q&A"
    )

    # ── How to use instructions ───────────────────────────────────────────
    with st.expander(
        "📖 How to use this tab — Best time to scan & what results mean", expanded=False
    ):
        st.markdown(
            """
<div style="padding:4px 0">

<div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#FF9A3C;margin-bottom:12px;">
  ⏰ Best Time to Scan
</div>

<table style="width:100%;border-collapse:collapse;font-size:.85rem;">
  <thead>
    <tr style="background:rgba(255,107,53,.2);">
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;border-radius:6px 0 0 0;">When you scan</th>
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">What it means</th>
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;border-radius:0 6px 0 0;">Recommended mode</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:white;font-weight:600;">🌅 Before market open (8–9 AM)</td>
      <td style="padding:10px 14px;color:#ccc;">Based on yesterday's close — plan today's entry/exit</td>
      <td style="padding:10px 14px;"><span style="background:rgba(156,39,176,.3);color:#ce93d8;padding:2px 10px;border-radius:20px;font-size:.75rem;">📅 End of Day</span></td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:white;font-weight:600;">⚡ During market hours (9:15 AM–3:30 PM)</td>
      <td style="padding:10px 14px;color:#ccc;">Near-live prices (~15 min delay) — intraday momentum snapshot</td>
      <td style="padding:10px 14px;"><span style="background:rgba(255,152,0,.2);color:#ffcc80;padding:2px 10px;border-radius:20px;font-size:.75rem;">⚡ Intraday</span></td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:white;font-weight:600;">🌆 After market close (4–9 PM)</td>
      <td style="padding:10px 14px;color:#ccc;"><b style="color:#00ff88">Best time</b> — full day data locked in, plan tomorrow's trades</td>
      <td style="padding:10px 14px;"><span style="background:rgba(156,39,176,.3);color:#ce93d8;padding:2px 10px;border-radius:20px;font-size:.75rem;">📅 End of Day</span></td>
    </tr>
    <tr>
      <td style="padding:10px 14px;color:white;font-weight:600;">📅 Weekend</td>
      <td style="padding:10px 14px;color:#ccc;">Weekly review — identify setups for next week</td>
      <td style="padding:10px 14px;"><span style="background:rgba(156,39,176,.3);color:#ce93d8;padding:2px 10px;border-radius:20px;font-size:.75rem;">📅 End of Day</span></td>
    </tr>
  </tbody>
</table>

<div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#FF9A3C;margin:20px 0 12px;">
  📊 What the Signals Mean
</div>

<table style="width:100%;border-collapse:collapse;font-size:.85rem;">
  <thead>
    <tr style="background:rgba(255,107,53,.2);">
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">Label</th>
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">Full name</th>
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">What it tells you</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">1W</td>
      <td style="padding:10px 14px;color:#ccc;">1 Week (5 trading days)</td>
      <td style="padding:10px 14px;color:#ccc;">Short-term price momentum</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">1M</td>
      <td style="padding:10px 14px;color:#ccc;">1 Month (21 trading days)</td>
      <td style="padding:10px 14px;color:#ccc;">Medium-term trend strength</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">3M</td>
      <td style="padding:10px 14px;color:#ccc;">3 Months (63 trading days)</td>
      <td style="padding:10px 14px;color:#ccc;">Longer trend — useful for swing trades</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">RSI</td>
      <td style="padding:10px 14px;color:#ccc;">Relative Strength Index (14-day)</td>
      <td style="padding:10px 14px;color:#ccc;">&lt;30 = oversold (buy zone) · &gt;70 = overbought (sell zone)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">52W pos</td>
      <td style="padding:10px 14px;color:#ccc;">52-week price position</td>
      <td style="padding:10px 14px;color:#ccc;">Where price sits between 52W low and high (%)</td>
    </tr>
    <tr>
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">Score</td>
      <td style="padding:10px 14px;color:#ccc;">Composite score (0–100)</td>
      <td style="padding:10px 14px;color:#ccc;">72+ = Strong Buy · 60+ = Buy · &lt;32 = Sell · &lt;20 = Strong Sell</td>
    </tr>
  </tbody>
</table>

<div style="margin-top:16px;padding:12px 16px;background:rgba(255,107,53,.1);border-left:3px solid #FF6B35;border-radius:0 8px 8px 0;font-size:.82rem;color:#ccc;">
  ⚠️ <b style="color:white">Disclaimer:</b> This is AI-powered technical analysis, not financial advice.
  Always do your own research and consult a SEBI-registered advisor before investing.
  Past patterns do not guarantee future results.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ── Session state ─────────────────────────────────────────────────────
    for _k, _v in [
        ("scan_results", {}),
        ("scan_done", False),
        ("chat_history", []),
        ("scan_universe", ""),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Controls ──────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([2, 1.5, 1.5])

    scan_universe = sc1.selectbox(
        "Scan Universe",
        [
            "📁 My Portfolio",
            "⭐ Nifty 50",
            "💯 Nifty 100",
            "🌐 Nifty 500",
            "🏦 Banking & Finance",
            "💻 IT & Technology",
            "🚗 Automobile & EV",
            "⚡ Energy & Power",
            "🧪 Chemicals & Pharma",
            "🛒 FMCG & Consumer",
            "🏥 Healthcare & Diagnostics",
            "🏗️ Infrastructure & Real Estate",
            "🔩 Metals & Mining",
            "🏭 Manufacturing & Industrials",
        ],
        key="scan_universe_sel",
    )
    top_n = sc2.slider("Top N picks", 5, 20, 10, key="scan_topn")
    scan_btn = sc3.button(
        "🚀 Run Scanner", type="primary", key="scan_run", use_container_width=True
    )

    # ── Scan Mode toggle ──────────────────────────────────────────────────
    sm1, sm2, sm3 = st.columns([1.5, 1.5, 5])
    scan_mode = sm1.radio(
        "Scan Mode",
        ["⚡ Intraday", "📅 End of Day"],
        index=0,
        key="scan_mode",
        horizontal=True,
        help="Intraday: weights RSI + Volume surge more heavily for quick trades.\n"
        "End of Day: weights MA alignment + Momentum for swing/positional trades.",
    )

    # Mode badge + explanation
    is_intraday = "Intraday" in scan_mode
    if is_intraday:
        sm2.markdown(
            '<div style="background:rgba(255,152,0,.15);border:1px solid #ff9800;'
            'border-radius:8px;padding:8px 12px;font-size:.78rem;color:#ffcc80;">'
            "⚡ <b>Intraday mode</b><br>"
            "RSI & Volume weighted 2× higher.<br>"
            "Best for same-day trades.</div>",
            unsafe_allow_html=True,
        )
    else:
        sm2.markdown(
            '<div style="background:rgba(156,39,176,.15);border:1px solid #9c27b0;'
            'border-radius:8px;padding:8px 12px;font-size:.78rem;color:#ce93d8;">'
            "📅 <b>End of Day mode</b><br>"
            "MA alignment & Momentum weighted 2× higher.<br>"
            "Best for swing & positional trades.</div>",
            unsafe_allow_html=True,
        )

    # ── Build ticker list for selected universe ────────────────────────────
    def _get_scan_stocks(choice: str) -> list[tuple[str, str]]:
        if "Portfolio" in choice:
            port = st.session_state.get("portfolio", [])
            if not port:
                return []
            return [
                (e.get("ticker", ""), e.get("company", ""))
                for e in port
                if e.get("ticker") and e.get("company")
            ]
        if "Nifty 500" in choice:
            from nse_stocks import fetch_nifty500

            return fetch_nifty500()
        if "Nifty 100" in choice:
            from nse_stocks import NIFTY100, NIFTY100_NAMES

            return [(t, NIFTY100_NAMES.get(t, t.replace(".NS", ""))) for t in NIFTY100]
        if "Nifty 50" in choice:
            return [(t, NIFTY50_NAMES.get(t, t.replace(".NS", ""))) for t in NIFTY50]
        for sk, stocks in NSE_STOCKS.items():
            if sk in choice or choice in sk:
                return [(f"{s}.NS", n) for s, n in stocks]
        return []

    # ── Run scan ──────────────────────────────────────────────────────────
    if scan_btn:
        stocks_to_scan = _get_scan_stocks(scan_universe)
        if not stocks_to_scan:
            st.warning("No stocks found. If using Portfolio, add stocks first.")
        else:
            st.info(
                f"Scanning {len(stocks_to_scan)} stocks — this takes ~{len(stocks_to_scan)//3}–{len(stocks_to_scan)//2} seconds..."
            )
            prog_bar = st.progress(0.0, text="Initializing scanner...")
            prog_text = st.empty()

            def _prog(pct, msg):
                prog_bar.progress(pct, text=msg)
                prog_text.caption(msg)

            with st.spinner("Scanner running..."):
                results = run_scanner(
                    stocks_to_scan,
                    top_n=top_n,
                    progress_callback=_prog,
                    mode="intraday" if is_intraday else "eod",
                )
            prog_bar.progress(
                1.0, text=f"✅ Scanned {results['total_scanned']} stocks!"
            )
            st.session_state.scan_results = results
            st.session_state.scan_done = True
            st.session_state.scan_universe = scan_universe
            st.session_state.scan_mode_lbl = scan_mode
            st.session_state.chat_history = []

    # ── Display results ───────────────────────────────────────────────────
    if st.session_state.scan_done and st.session_state.scan_results:
        results = st.session_state.scan_results
        buys = results.get("buys", [])
        sells = results.get("sells", [])
        all_r = results.get("all", [])

        st.markdown("---")

        # ── Summary metrics ───────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Stocks Scanned", results["total_scanned"])
        m2.metric("🟢 Buy Signals", len([r for r in all_r if "BUY" in r["direction"]]))
        m3.metric(
            "🔴 Sell Signals", len([r for r in all_r if "SELL" in r["direction"]])
        )
        m4.metric("⬜ Neutral", len([r for r in all_r if r["direction"] == "NEUTRAL"]))
        avg_score = (
            round(sum(r["score"] for r in all_r) / len(all_r), 1) if all_r else 0
        )
        m5.metric("Avg Score", f"{avg_score}/100")

        st.markdown(
            f"*Scan completed: {results['scan_time']} · Universe: {st.session_state.scan_universe} · Mode: {st.session_state.get('scan_mode_lbl','—')}*"
        )
        st.markdown("---")

        # ── BUY and SELL columns ──────────────────────────────────────
        buy_col, sell_col = st.columns(2)

        def _score_bar(score, is_buy):
            bar_cls = "rec-score-bar-g" if is_buy else "rec-score-bar-r"
            pct = score
            return (
                f'<div class="rec-score-bar-bg">'
                f'<div class="{bar_cls}" style="width:{pct}%"></div></div>'
            )

        def _breakdown_pills(bd: dict) -> str:
            labels = {
                "rsi": "RSI",
                "ma": "MA",
                "momentum": "Mom",
                "volume": "Vol",
                "pos52": "52W",
                "pattern": "Pattern",
            }
            pills = ""
            for k, v in bd.items():
                pills += f'<span class="rec-pill">{labels.get(k,k)}: {v}</span>'
            return pills

        with buy_col:
            st.markdown(
                '<div class="board-hdr-g">🟢 TOP BUY RECOMMENDATIONS</div>',
                unsafe_allow_html=True,
            )
            if not buys:
                st.info("No strong buy signals found in this universe.")
            for i, s in enumerate(buys):
                is_strong = "STRONG" in s["direction"]
                chg_sign = "+" if s["today_change"] >= 0 else ""
                tgt_pct = (s["target"] - s["current_price"]) / s["current_price"] * 100

                st.markdown(
                    f"""
<div class="rec-card-buy">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="rec-rank">#{i+1}</span>
    <div>
      <div class="rec-name">{s["strength"]} {s["name"]}</div>
      <div class="rec-ticker">{s["ticker"]} · {s["direction"]}</div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div class="rec-price">₹{s["current_price"]:,.2f}</div>
      <div style="font-size:.8rem;color:{'#00ff88' if s['today_change']>=0 else '#ff6666'}">
        {chg_sign}{s["today_change"]:.2f}% today
      </div>
    </div>
  </div>
  <div style="font-size:.78rem;color:#aaa;margin-bottom:4px;">
    Score: <b style="color:#00ff88">{s["score"]}/100</b>
  </div>
  {_score_bar(s["score"], True)}
  <div class="rec-meta" style="margin-top:8px;">
    🎯 Target: <b>₹{s["target"]:,.2f}</b> (+{tgt_pct:.1f}%) &nbsp;|&nbsp;
    🛑 Stop Loss: <b>₹{s["stop_loss"]:,.2f}</b>
  </div>
  <div class="rec-meta">
    RSI: {s["rsi"] or "—"} &nbsp;|&nbsp;
    MA20: {'₹'+str(s["ma20"]) if s["ma20"] else "—"} &nbsp;|&nbsp;
    Pattern: {s["primary_pattern"]} &nbsp;|&nbsp;
    {'🏆 Golden Cross!' if s["golden_cross"] else ''}
    1M: {('+' if (s["ret_1m"] or 0)>=0 else '')+str(s["ret_1m"])+'%' if s["ret_1m"] else "—"}
  </div>
  <div class="rec-breakdown">{_breakdown_pills(s["score_breakdown"])}</div>
</div>""",
                    unsafe_allow_html=True,
                )

        with sell_col:
            st.markdown(
                '<div class="board-hdr-r">🔴 TOP SELL / AVOID</div>',
                unsafe_allow_html=True,
            )
            if not sells:
                st.info("No strong sell signals found in this universe.")
            for i, s in enumerate(sells):
                chg_sign = "+" if s["today_change"] >= 0 else ""
                tgt_pct = (s["target"] - s["current_price"]) / s["current_price"] * 100

                st.markdown(
                    f"""
<div class="rec-card-sell">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <span class="rec-rank">#{i+1}</span>
    <div>
      <div class="rec-name">{s["strength"]} {s["name"]}</div>
      <div class="rec-ticker">{s["ticker"]} · {s["direction"]}</div>
    </div>
    <div style="margin-left:auto;text-align:right;">
      <div class="rec-price">₹{s["current_price"]:,.2f}</div>
      <div style="font-size:.8rem;color:{'#00ff88' if s['today_change']>=0 else '#ff6666'}">
        {chg_sign}{s["today_change"]:.2f}% today
      </div>
    </div>
  </div>
  <div style="font-size:.78rem;color:#aaa;margin-bottom:4px;">
    Score: <b style="color:#ff6666">{s["score"]}/100</b>
  </div>
  {_score_bar(100 - s["score"], False)}
  <div class="rec-meta" style="margin-top:8px;">
    🎯 Target: <b>₹{s["target"]:,.2f}</b> ({tgt_pct:.1f}%) &nbsp;|&nbsp;
    🛑 Stop Loss: <b>₹{s["stop_loss"]:,.2f}</b>
  </div>
  <div class="rec-meta">
    RSI: {s["rsi"] or "—"} &nbsp;|&nbsp;
    MA20: {'₹'+str(s["ma20"]) if s["ma20"] else "—"} &nbsp;|&nbsp;
    Pattern: {s["primary_pattern"]} &nbsp;|&nbsp;
    {'💀 Death Cross!' if s["death_cross"] else ''}
    1M: {('+' if (s["ret_1m"] or 0)>=0 else '')+str(s["ret_1m"])+'%' if s["ret_1m"] else "—"}
  </div>
  <div class="rec-breakdown">{_breakdown_pills(s["score_breakdown"])}</div>
</div>""",
                    unsafe_allow_html=True,
                )

        # ── Full ranked table ─────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Full Ranked List — All Scanned Stocks", expanded=False):
            if all_r:
                df_all = pd.DataFrame(
                    [
                        {
                            "Rank": i + 1,
                            "Stock": r["name"],
                            "Ticker": r["ticker"],
                            "Score": r["score"],
                            "Signal": r["direction"],
                            "Price (₹)": r["current_price"],
                            "Today %": r["today_change"],
                            "RSI": r["rsi"],
                            "Target ₹": r["target"],
                            "Stop Loss": r["stop_loss"],
                            "Pattern": r["primary_pattern"],
                            "1M %": r["ret_1m"],
                        }
                        for i, r in enumerate(all_r)
                    ]
                )

                def _color_signal(v):
                    if "STRONG BUY" in str(v):
                        return "background-color:#0a5c2a;color:white"
                    if "BUY" in str(v):
                        return "background-color:#0d7a38;color:white"
                    if "STRONG SELL" in str(v):
                        return "background-color:#5c0d0d;color:white"
                    if "SELL" in str(v):
                        return "background-color:#7a1a1a;color:white"
                    return "background-color:#333;color:white"

                st.dataframe(
                    df_all.style.applymap(_color_signal, subset=["Signal"]),
                    use_container_width=True,
                    height=400,
                )
                st.download_button(
                    "⬇️ Download Recommendations CSV",
                    data=df_all.to_csv(index=False),
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        # ── AI CHATBOT ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<p class="section-title">🤖 Ask the AI Advisor</p>', unsafe_allow_html=True
        )
        st.caption(
            "Ask anything about the scan results — 'Why is X recommended?', 'Compare TCS vs Infosys', 'What is RSI?'"
        )

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(
                    f'<div class="{cls}">{icon} {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Suggested questions
        st.markdown("**Quick questions:**")
        sq_cols = st.columns(4)
        suggestions = [
            "Why is the top stock a BUY?",
            "What is RSI and how does it affect scores?",
            "Which stock has the best risk-reward?",
            "Explain the score breakdown",
        ]
        for i, q in enumerate(suggestions):
            if sq_cols[i].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state._pending_chat = q

        # Chat input
        user_input = st.chat_input("Ask about any stock or the recommendations...")

        # Handle pending suggestion click
        if (
            hasattr(st.session_state, "_pending_chat")
            and st.session_state._pending_chat
        ):
            user_input = st.session_state._pending_chat
            st.session_state._pending_chat = None

        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            with st.spinner("AI advisor thinking..."):
                reply = chat_with_advisor(
                    user_input,
                    st.session_state.scan_results,
                    st.session_state.chat_history[:-1],
                )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()

        if len(st.session_state.chat_history) > 0:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # No scan done yet — show instructions
        st.markdown(
            """
<div style="text-align:center;padding:60px 20px;color:#888;">
  <div style="font-size:3rem;margin-bottom:16px;">🎯</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#ccc;margin-bottom:8px;">
    Ready to scan
  </div>
  <div style="font-size:.9rem;max-width:500px;margin:0 auto;line-height:1.6;">
    Select a universe above (your portfolio, Nifty 50, Nifty 500, or any sector)
    and click <b>🚀 Run Scanner</b>.<br><br>
    The AI will score every stock on 6 signals — RSI, MA alignment, momentum,
    volume surge, 52W position, and Bulkowski chart patterns —
    then rank the best <b>BUY</b> and <b>SELL/AVOID</b> picks for you.
  </div>
</div>""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 6 — RL TRADING SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown(
        '<p class="section-title">🤖 RL Trading Simulator</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Train a Reinforcement Learning agent (PPO / A2C / DQN) on historical NSE data "
        "and backtest against Buy & Hold • Powered by Stable-Baselines3 & Gymnasium"
    )

    # ── How it works ──────────────────────────────────────────────────────
    with st.expander("📖 How this works — RL Trading explained", expanded=False):
        st.markdown(
            """
<div style="padding:4px 0">

<div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#FF9A3C;margin-bottom:12px;">
  🧠 What is RL Trading?
</div>

<div style="font-size:.88rem;color:rgba(255,255,255,.85);line-height:1.7;margin-bottom:16px;">
  A <b>Reinforcement Learning (RL)</b> agent learns a trading policy by interacting with historical
  market data. It observes technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
  and decides to <b>BUY</b>, <b>HOLD</b>, or <b>SELL</b> at each time step.
  <br><br>
  The agent is rewarded for portfolio growth and penalised for drawdowns. Over thousands of
  training episodes, it discovers patterns and timing strategies that maximize returns.
</div>

<table style="width:100%;border-collapse:collapse;font-size:.85rem;">
  <thead>
    <tr style="background:rgba(255,107,53,.2);">
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">Component</th>
      <th style="padding:10px 14px;text-align:left;color:#FF9A3C;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">📊 Observations</td>
      <td style="padding:10px 14px;color:#ccc;">RSI, MACD, Bollinger Width, MA ratios, volume ratio, 5D return, position & P&L</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">🎯 Actions</td>
      <td style="padding:10px 14px;color:#ccc;">BUY (go all-in), HOLD (do nothing), SELL (exit position)</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">💰 Reward</td>
      <td style="padding:10px 14px;color:#ccc;">Log-return of portfolio value + drawdown penalty + trade cost penalty</td>
    </tr>
    <tr>
      <td style="padding:10px 14px;color:#FFD700;font-weight:700;">🔀 Train/Test</td>
      <td style="padding:10px 14px;color:#ccc;">80% training (agent learns), 20% testing (out-of-sample backtest)</td>
    </tr>
  </tbody>
</table>

<div style="margin-top:16px;padding:12px 16px;background:rgba(255,107,53,.1);border-left:3px solid #FF6B35;border-radius:0 8px 8px 0;font-size:.82rem;color:#ccc;">
  ⚠️ <b style="color:white">Disclaimer:</b> This is an educational simulator. RL trading models trained on
  limited data may overfit. Do NOT use for real trading without extensive validation.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ── Configuration ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.05rem;font-weight:700;'
        'color:#FF9A3C;border-left:4px solid #FF9A3C;padding-left:12px;margin:16px 0 14px;">'
        '⚙️ Configuration</div>',
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3, rc4 = st.columns(4)

    with rc1:
        from nse_stocks import NIFTY50, NIFTY50_NAMES

        rl_ticker_options = {v: k for k, v in NIFTY50_NAMES.items()}
        rl_ticker_name = st.selectbox(
            "📈 Select Stock",
            list(rl_ticker_options.keys()),
            index=list(rl_ticker_options.keys()).index("Reliance"),
            key="rl_ticker_select",
        )
        rl_ticker = rl_ticker_options[rl_ticker_name]

    with rc2:
        rl_period = st.selectbox(
            "📅 Historical Period",
            ["1y", "2y", "5y"],
            index=1,
            format_func=lambda x: {"1y": "1 Year", "2y": "2 Years", "5y": "5 Years"}[x],
            key="rl_period",
        )

    with rc3:
        rl_algo = st.selectbox(
            "🧠 Algorithm",
            ["PPO", "A2C", "DQN"],
            index=0,
            key="rl_algo",
            help="PPO is recommended. DQN is better for discrete actions. A2C is faster but noisier.",
        )

    with rc4:
        rl_timesteps = st.select_slider(
            "🔄 Training Steps",
            options=[5_000, 10_000, 20_000, 30_000, 50_000],
            value=10_000,
            key="rl_timesteps",
            help="More steps = better policy but slower training.",
        )

    rc5, rc6, rc7, _ = st.columns([1, 1, 1, 1])
    with rc5:
        rl_balance = st.number_input(
            "💰 Initial Balance (₹)",
            min_value=10_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            key="rl_balance",
        )
    with rc6:
        rl_fee = st.slider(
            "📊 Transaction Fee %",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            format="%.2f%%",
            key="rl_fee",
        )
    with rc7:
        rl_custom_ticker = st.text_input(
            "🔤 Or Custom Ticker",
            placeholder="e.g. TATAMOTORS.NS",
            key="rl_custom_ticker",
        )

    final_ticker = rl_custom_ticker.strip().upper() if rl_custom_ticker.strip() else rl_ticker

    # ── Run button ────────────────────────────────────────────────────────
    st.markdown("")
    run_col, info_col, _ = st.columns([1.5, 3, 2])
    run_rl = run_col.button(
        "🚀 Run Simulation", type="primary", key="rl_run_btn", use_container_width=True
    )
    info_col.markdown(
        f'<div style="font-size:.82rem;color:#aaa;padding-top:8px;">'
        f'Will train <b style="color:#FFD700">{rl_algo}</b> on '
        f'<b style="color:#00ff88">{final_ticker}</b> for '
        f'<b style="color:#FF9A3C">{rl_timesteps:,}</b> steps '
        f'(₹{rl_balance:,.0f} balance, {rl_fee:.2f}% fee)</div>',
        unsafe_allow_html=True,
    )

    # ── Session state for results ─────────────────────────────────────────
    if "rl_results" not in st.session_state:
        st.session_state.rl_results = None

    if run_rl:
        st.markdown("---")
        prog_bar = st.progress(0.0, text="Initializing...")
        log_placeholder = st.empty()
        log_msgs = []

        def _rl_progress(pct, msg):
            prog_bar.progress(min(pct, 1.0), text=msg)
            ts = datetime.now().strftime("%H:%M:%S")
            log_msgs.append(f"[{ts}] {msg}")
            log_placeholder.markdown(
                f'<div class="status-log">{"<br>".join(log_msgs[-6:])}</div>',
                unsafe_allow_html=True,
            )

        try:
            from rl_simulator import train_and_backtest

            results = train_and_backtest(
                ticker=final_ticker,
                period=rl_period,
                algorithm=rl_algo,
                timesteps=rl_timesteps,
                initial_balance=float(rl_balance),
                fee_pct=rl_fee / 100.0,
                progress_callback=_rl_progress,
            )
            st.session_state.rl_results = results
            prog_bar.progress(1.0, text="✅ Simulation complete!")
            st.success(
                f"Training complete! {rl_algo} agent trained on {results['train_days']} days, "
                f"backtested on {results['test_days']} days."
            )

        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.session_state.rl_results = None

    # ── Display results ───────────────────────────────────────────────────
    results = st.session_state.rl_results
    if results:
        metrics = results["metrics"]

        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'Syne\',sans-serif;font-size:1.05rem;font-weight:700;'
            'color:#FF9A3C;border-left:4px solid #FF9A3C;padding-left:12px;margin:16px 0 14px;">'
            f'📊 Results — {results["ticker"]} ({results["algorithm"]}, '
            f'{results["timesteps"]:,} steps)</div>',
            unsafe_allow_html=True,
        )

        # ── Metric cards ──────────────────────────────────────────────
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)

        # Determine colors
        rl_color = "#00ff88" if metrics["rl_return"] >= 0 else "#ff4444"
        bh_color = "#00ff88" if metrics["bh_return"] >= 0 else "#ff4444"
        alpha_color = "#00ff88" if metrics["alpha"] >= 0 else "#ff4444"

        mc1.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">🤖 RL Return</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{rl_color};margin-top:4px;">
            {metrics['rl_return']:+.2f}%</div>
            <div style="font-size:.72rem;color:#888;margin-top:2px;">₹{metrics['final_value']:,.0f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">📈 Buy & Hold</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{bh_color};margin-top:4px;">
            {metrics['bh_return']:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        mc3.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">⚡ Alpha</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{alpha_color};margin-top:4px;">
            {metrics['alpha']:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        mc4.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">📐 Sharpe Ratio</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#FFD700;margin-top:4px;">
            {metrics['sharpe_ratio']:.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        mc5.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">📉 Max Drawdown</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#ff6666;margin-top:4px;">
            -{metrics['max_drawdown']:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        mc6.markdown(
            f"""<div style="background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:1px;">🔄 Trades / Win %</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#FF9A3C;margin-top:4px;">
            {metrics['total_trades']} / {metrics['win_rate']:.0f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Alpha verdict ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if metrics["alpha"] > 2:
            verdict_html = (
                '<div style="background:rgba(0,200,83,.1);border:1px solid rgba(0,200,83,.4);'
                'border-radius:12px;padding:14px 20px;border-left:5px solid #00c853;">'
                '<span style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.1rem;color:#00ff88;">'
                f'🏆 RL Agent BEAT Buy & Hold by {metrics["alpha"]:+.2f}%</span>'
                '<br><span style="font-size:.82rem;color:#aaa;">The trained policy found profitable trading opportunities '
                'that outperformed passive investing.</span></div>'
            )
        elif metrics["alpha"] > -2:
            verdict_html = (
                '<div style="background:rgba(255,214,0,.08);border:1px solid rgba(255,214,0,.3);'
                'border-radius:12px;padding:14px 20px;border-left:5px solid #ffd600;">'
                '<span style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.1rem;color:#ffd600;">'
                f'🤝 RL Agent performed SIMILAR to Buy & Hold ({metrics["alpha"]:+.2f}% alpha)</span>'
                '<br><span style="font-size:.82rem;color:#aaa;">The agent matched the benchmark. '
                'Try more training steps or a different algorithm.</span></div>'
            )
        else:
            verdict_html = (
                '<div style="background:rgba(255,23,68,.08);border:1px solid rgba(255,23,68,.3);'
                'border-radius:12px;padding:14px 20px;border-left:5px solid #ff1744;">'
                '<span style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.1rem;color:#ff4444;">'
                f'📉 RL Agent UNDERPERFORMED Buy & Hold by {abs(metrics["alpha"]):.2f}%</span>'
                '<br><span style="font-size:.82rem;color:#aaa;">The model needs more training steps, '
                'a different algorithm, or a longer data period.</span></div>'
            )
        st.markdown(verdict_html, unsafe_allow_html=True)

        # ── Equity Curve Chart ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:700;'
            'color:white;margin-bottom:8px;">📈 Equity Curve — RL Agent vs Buy & Hold</div>',
            unsafe_allow_html=True,
        )

        equity = results["equity_curve"]
        bh_curve = results["buy_hold_curve"]
        dates = results["test_dates"]

        # Ensure all arrays match length
        min_len = min(len(equity), len(bh_curve), len(dates))
        equity = equity[:min_len]
        bh_curve = bh_curve[:min_len]
        dates = dates[:min_len]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                name="🤖 RL Agent",
                line=dict(color="#00ff88", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.05)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bh_curve,
                name="📈 Buy & Hold",
                line=dict(color="#FF9A3C", width=2, dash="dash"),
            )
        )

        # Add trade markers
        buy_trades = [t for t in results["trades"] if t["action"] == "BUY"]
        sell_trades = [t for t in results["trades"] if "SELL" in t["action"]]

        if buy_trades:
            buy_steps = [t["step"] for t in buy_trades if t["step"] < min_len]
            buy_vals = [equity[s] for s in buy_steps]
            buy_dates = [dates[s] for s in buy_steps]
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_vals,
                    mode="markers",
                    name="🟢 Buy",
                    marker=dict(
                        color="#00ff88",
                        size=10,
                        symbol="triangle-up",
                        line=dict(color="white", width=1),
                    ),
                )
            )

        if sell_trades:
            sell_steps = [t["step"] for t in sell_trades if t["step"] < min_len]
            sell_vals = [equity[s] for s in sell_steps]
            sell_dates = [dates[s] for s in sell_steps]
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_vals,
                    mode="markers",
                    name="🔴 Sell",
                    marker=dict(
                        color="#ff4444",
                        size=10,
                        symbol="triangle-down",
                        line=dict(color="white", width=1),
                    ),
                )
            )

        # Add horizontal line at initial balance
        fig.add_hline(
            y=results["initial_balance"],
            line_dash="dot",
            line_color="rgba(255,255,255,0.2)",
            annotation_text=f"Initial ₹{results['initial_balance']:,.0f}",
            annotation_position="bottom right",
            annotation_font_color="#888",
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(family="Syne, sans-serif"),
            height=450,
            margin=dict(l=60, r=40, t=30, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            yaxis=dict(
                title="Portfolio Value (₹)",
                gridcolor="rgba(255,255,255,0.05)",
                tickformat=",.0f",
                tickprefix="₹",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Trade Log ─────────────────────────────────────────────────
        st.markdown(
            '<div style="font-family:\'Syne\',sans-serif;font-size:1rem;font-weight:700;'
            'color:white;margin:16px 0 8px;">📋 Trade Log</div>',
            unsafe_allow_html=True,
        )

        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            # Style the dataframe
            display_cols = [c for c in ["date", "action", "price", "shares", "value", "pnl_pct"] if c in trades_df.columns]
            trades_display = trades_df[display_cols].copy()
            trades_display.columns = [
                {"date": "Date", "action": "Action", "price": "Price (₹)",
                 "shares": "Shares", "value": "Portfolio (₹)", "pnl_pct": "P&L %"}.get(c, c)
                for c in display_cols
            ]
            st.dataframe(
                trades_display,
                use_container_width=True,
                hide_index=True,
                height=min(400, 40 + len(trades_display) * 35),
            )
        else:
            st.info("No trades were executed by the agent.")

    elif not run_rl:
        # No results yet — show instructions
        st.markdown(
            """
<div style="text-align:center;padding:60px 20px;color:#888;">
  <div style="font-size:3rem;margin-bottom:16px;">🤖</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#ccc;margin-bottom:8px;">
    Ready to train
  </div>
  <div style="font-size:.9rem;max-width:550px;margin:0 auto;line-height:1.6;">
    Select a stock and configure the RL agent above, then click
    <b>🚀 Run Simulation</b>.<br><br>
    The AI will train a neural-network trading policy on 80% of historical data,
    then backtest on the remaining 20% — completely out of sample.<br><br>
    <span style="color:#FF9A3C;">💡 Tip:</span> Start with <b>PPO</b>, <b>10,000 steps</b>,
    and <b>2 Years</b> of data for best results.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
